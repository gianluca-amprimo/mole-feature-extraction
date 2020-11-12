# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:26 2019

@author: Gianluca
"""


import numpy as np
import math
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.image as im

def quant_image(image, n_clust=3): #default is using 3 clusters
 
    #quantization 
    kmeans=KMeans(n_clusters=n_clust, random_state=0) 
    [N1,N2, N3]=image.shape
    im_2D=image.reshape((N1*N2, N3))
    kmeans.fit(im_2D)
    centroids=kmeans.cluster_centers_.astype('uint8')
    quant_im=kmeans.labels_.reshape((N1, N2))
        
    return quant_im, centroids


def median_filter(img, size): #the filter is applied only where it perfectly fits, no padding is used (border will not be filtered because it should contain mole pixel)
    N1, N2=img.shape
    radius=int(size/2)
    filtered_image=np.ones([N1, N2])
    for i in range(N1):
        for j in range(N2):
            if(i-radius<0 or j-radius<0 or i+radius>=N1 or j+radius>=N2): #does not fit
                filtered_image[i,j]=img[i,j]
            else: #it fits
                kernel=img[i-radius:i+radius+1,j-radius:j+radius+1]
                sorted_list=np.sort(kernel, axis=None)
                filtered_image[i,j]=sorted_list[int(size**2/2)]
    return filtered_image
        
def crop(image, x_max, x_min, y_max, y_min): #crop with a margin of 20 pixel each side
    N1=image.shape[0]
    N1_new=x_max-x_min+40
    N2_new=y_max-y_min+40
    if len(image.shape)==3:   
        new_image=np.zeros((N1_new,N2_new,3))
        new_image[:,:,0]=image[(N1-1-x_max)-20:(N1-1-x_min)+20, y_min-20:y_max+20, 0]
        new_image[:,:,1]=image[(N1-1-x_max)-20:(N1-1-x_min)+20, y_min-20:y_max+20, 1]
        new_image[:,:,2]=image[(N1-1-x_max)-20:(N1-1-x_min)+20, y_min-20:y_max+20, 2]
    else:        
        new_image=np.zeros((N1_new,N2_new))
        new_image=image[(N1-1-x_max)-20:(N1-1-x_min)+20, y_min-20:y_max+20]
    return new_image


#for edge detection use Sobel filter 
def sobel_filter(image):
    N1, N2=image.shape
    edges=np.ones([N1,N2])
    Gx_filter=np.array([[-1,0,+1], [-2,0,+2], [-1, 0, +1]])
    Gy_filter=np.array([[+1,+2,+1],[0,0,0],[-1,-2,-1]])
    
    for i in range(N1):
        for j in range(N2):
            image_sample=np.zeros([3,3])
            if (i-1>0 and j-1>0 and i+1<N1 and j+1<N2): #filter is completely over image
                image_sample=image[i-1:i+2,j-1:j+2]
            if(i-1<0 and j-1>0 and j+1<N2):
                image_sample[1:,:]=image[i:i+2,j-1:j+2]
            if(i-1<0 and j-1<0):
                image_sample[1:,1:]=image[i:i+2,j:j+2]
            if(i-1>0 and j-1<0 and i+1<N1):
                 image_sample[:, 1:]=image[i-1:i+2, j:j+2]
            if(i+1>=N1 and j-1<0):
                 image_sample[:2, 1:]=image[i-1:i+1, j:j+2]
            if(i+1>=N1 and j+1<N2 and j-1>=0):
                 image_sample[:2,:]=image[i-1:i+1, j-1:j+2]
            if(i+1>=N1 and j+1>=N2):
                 image_sample[:2, :2]=image[i-1:i+1, j-1:j+1]
            if(i+1<N1  and j+1>=N2 and i-1>=0):
                 image_sample[:, :2]=image[i-1:i+2, j-1:j+1]
            if(i-1<0 and j+1>=N2):
                 image_sample[:2, :2]=image[i:i+2, j-1:j+1]
         
            Gx=(image_sample*Gx_filter).sum()
            Gy=(image_sample*Gy_filter).sum()
            G=math.sqrt(Gx**2+Gy**2)
            edges[i,j]=G
   
    edges=((np.logical_not(edges>2.8)).astype('uint8')) #thresh 2.8
    edges[:N1,0]=1
    edges[:N1,N2-1]=1
    edges[0,:]=1
    edges[N1-1,:]=1
    
    return edges

def check_color(value):
    if value==1:
        return True
    else:
        return False
            
def flood_fill(image,centre): #due to stack overflow, I implement the non recursive version
    
    x=int(centre[0])
    y=int(centre[1])
    queue=[]
    queue.insert(0,(x,y))
    while len(queue)!=0:
       x1,y1 = queue.pop()
       if image[x1, y1]!=0:
           image[x1, y1]=0
           if check_color(image[x1+1,y1]) :
                queue.insert(0,(x1+1,y1))
           if check_color(image[x1-1,y1]) :
                queue.insert(0,(x1-1,y1))
           if check_color(image[x1,y1+1]) :
                queue.insert(0,(x1,y1+1))
           if check_color(image[x1,y1-1]) :
                queue.insert(0,(x1,y1-1))         
    return image
    
def mole_center(coordinates):
    centre=np.array((1,2))
    centre[0]=np.mean(coordinates[:,0])
    centre[1]=np.mean(coordinates[:,1])
    return centre  #order is y, x

def mole_by_coord(molebn):
    N1,N2=molebn.shape
    X=[]
    for i in range(N1):
        for j in range(N2):
            if molebn[i][j]==0:
                X.append([N1-1-i,j])
    return X

def apply_blur(image, size=7): #applies median filter n_times to produce a blurred image, default is 3 times
    N1,N2,N3=image.shape
    image_filt=np.zeros((N1,N2,N3))
    image_filt[:,:,0]=median_filter(image[:,:,0],size)
    image_filt[:,:,1]=median_filter(image[:,:,1],size)
    image_filt[:,:,2]=median_filter(image[:,:,2],size)
    image_filt=image_filt.astype('uint8')
    return image_filt

#idea: try again a spatial clustering with DBSCAN, maintain only the bigger cluster that should be the countour of the mole
def remove_holes(X, image):
    
    N1, N2=image.shape
    x=X[:,1]
    y=X[:,0]
    
    # Compute DBSCAN
    db = DBSCAN(eps=3, min_samples=2).fit(X) #general value 6, 10 solves some issues for melanoma_10
    labels = db.labels_
    labels+=2 #trick to avoid labels 0 and 1 that are already value of image pixel
    
    unique_labels = set(labels)
    biggest_contour=(0,0)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    fig, ax = plt.subplots()
    i=0
    for l in unique_labels:
        ix = np.where(labels == l)
        ax.scatter(x[ix], y[ix], c = colors[i], label = l)
        perimeter=np.count_nonzero(labels==l)
        if(perimeter>biggest_contour[1]):
            biggest_contour=(l, perimeter)
        i+=1
    ax.legend()
    plt.title("DBSCAN results for mole holes")
    plt.pause(0.1)
    n=0
    for r in X:
        image[N1-1-r[0]][r[1]]=labels[n]
        n+=1
        
    clean_image=((image!=biggest_contour[0]).astype('uint8'))
    
    return clean_image, biggest_contour[1] 

#idea: apply DBSCAN but based on spatial parameters, not colour
def spatial_filtering(X, pic_center): #function receives coordinates of points prev classified as mole, onyl cluster closer to image center is maintained
    
    x=X[:,1]
    y=X[:,0]
    
    # Compute DBSCAN
    db = DBSCAN(eps=5, min_samples=15).fit(X) 
    labels = db.labels_
    labels+=2 #trick to avoid labels 0 and 1 that are already values of image pixels
    
    unique_labels = set(labels)
    closest_label=(0 ,float('inf'))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    fig, ax = plt.subplots()
    i=0
    for l in unique_labels:
        ix = np.where(labels == l)
        ax.scatter(x[ix], y[ix], c = colors[i], label = l)
        cluster=np.array([x[ix], y[ix]]).T
        center=mole_center(cluster)
        distance=np.linalg.norm(pic_center-center)
        if(distance<closest_label[1]):
            closest_label=(l, distance)
        i=i+1
    ax.legend()
    plt.title("DBSCAN for external noise")
    plt.pause(0.1)
    
    return labels, closest_label[0]
    
def indentation(image, perimeter):
    equivalent_area=np.count_nonzero(image==0)
    print("Area is ",equivalent_area)
    ideal_perim=math.sqrt(equivalent_area/math.pi)*2*math.pi
    print("Ideal perimeter is", ideal_perim)
    return (perimeter/2)/ideal_perim

def symmetry(X, x_c, y_c, im_name):
    #X is a list of (y,x) y=image row, x=image column
    best_symmetry=(0,0)
    
    X_vec=np.array(X)
    i=0
    for vertical_inc in range(0,180,20):
        left_matches=0
        right_matches=0
        left_points=0
        right_points=0
        match_list_left=[]
        match_list_right=[]
        if(vertical_inc==0):
            m='inf'
        else:
            m=math.tan(math.radians(90-vertical_inc)) #slope of symmetry axis
       
        #make symmetry of mole points according to current axis    
        for r in X:
            x_p=r[1]
            y_p=r[0]
            if(m=='inf'):   #vertical axis has an equation x-x_c=0
                if(x_p-x_c==0): #point on the axis count both for left and right
                    left_matches+=1 
                    left_points+=1
                    right_matches+=1
                    right_points+=1
                    match_list_left.append([x_p,y_p])
                    match_list_right.append([x_p,y_p])
                    
                if(x_p-x_c<0): #make symmetry of points on the left
                    #symmetric point
                    x_p1=x_c+(x_c-x_p) 
                    y_p1=y_p
                    #is it a mole point?
                    if([y_p1,x_p1] in X):
                        left_matches+=1
                        left_points+=1
                        match_list_left.append([x_p,y_p])
                    else:
                        left_points+=1
                else: #pixel on the right
                    x_p1=x_c-(x_p-x_c) 
                    y_p1=y_p
                    #is it a mole point?
                    if([y_p1,x_p1] in X):
                        right_matches+=1
                        right_points+=1
                        match_list_right.append([x_p,y_p])
                    else:
                        right_points+=1
                    
            else: #non vertical axis
                if((m*(x_p-x_c)+y_c-y_p)==0): #point on the axis
                    left_matches+=1
                    left_points+=1
                    right_matches+=1
                    right_points+=1
                    match_list_left.append([x_p,y_p])
                    match_list_right.append([x_p,y_p])
                if((m*(x_p-x_c)+y_c-y_p)<0): #left side
                    x_i=int(((x_p+(m**2)*x_c+m*(y_p-y_c))/(1+m**2)) + 0.5) #+0.5 to round
                    y_i=int(m*(x_i-x_c)+y_c+0.5)  #+0.5 to round
                    x_p1=2*x_i-x_p
                    y_p1=2*y_i-y_p
                    #is it a mole point?
                    if([y_p1,x_p1] in X):
                        left_matches+=1
                        left_points+=1
                        match_list_left.append([x_p,y_p])
                    else:
                        left_points+=1
                else: #right
                    x_i=int(((x_p+(m**2)*x_c+m*(y_p-y_c))/(1+m**2)) + 0.5) #+0.5 to round
                    y_i=int(m*(x_i-x_c)+y_c+0.5)  #+0.5 to round
                    x_p1=2*x_i-x_p
                    y_p1=2*y_i-y_p
                    #is it a mole point?
                    if([y_p1,x_p1] in X):
                        right_matches+=1
                        right_points+=1
                        match_list_right.append([x_p,y_p])
                    else:
                        right_points+=1
       
        left_sym=left_matches/float(left_points)
        right_sym=right_matches/float(right_points)
        if(left_sym<right_sym): #among the two, choose the minimum
            sym=left_sym
        else:
            sym=right_sym
        if(best_symmetry[1]<sym): #is this axis a better axis of symmetry?
            best_symmetry=(m, sym)
            best_left=match_list_left
            best_right=match_list_right
            
#uncomment to save in folder results of each iteration of the algorithm            
#        match_vec_left=np.array(match_list_left)
#        match_vec_right=np.array(match_list_right)
#        fig, ax = plt.subplots()
#        ax.scatter(X_vec[:,1], X_vec[:,0], c='b', marker='o')
#        ax.scatter(match_vec_left[:,0], match_vec_left[:,1], c='r', marker='o')
#        ax.scatter(match_vec_right[:,0], match_vec_right[:,1], c='y', marker='o')
#        title="Axis with m={} and symmetry={}".format(m, sym)
#        name="Iteration {}".format(i)
#        plt.title(title)
#        ax.plot(x_c, y_c, 'go')
#        plt.savefig(name)
#        plt.close(fig)  
#        i+=1
    
    #draw the best axis
    match_vec_left=np.array(best_left)
    match_vec_right=np.array(best_right)
    fig, ax = plt.subplots()
    ax.scatter(X_vec[:,1], X_vec[:,0], c='b', marker='o')
    ax.scatter(match_vec_left[:,0], match_vec_left[:,1], c='r', marker='o')
    ax.scatter(match_vec_right[:,0], match_vec_right[:,1], c='y', marker='o')
    title="Axis with m={} and symmetry={}".format(best_symmetry[0], round(best_symmetry[1],3))
    name="./fig/Best_axis_for_{}".format(im_name)
    plt.title(title)
    ax.plot(x_c, y_c, 'go')
    plt.savefig(name)
    plt.pause(0.1)  
    return  best_symmetry

if __name__=='__main__':
    n_images={'low_risk': 11, 'medium_risk':16, 'melanoma':27}
    plt.close("all")
    #the following dictionaries contain special parameters setting for number of cluster and blur of tough images
    n_clusters={'low_risk_10':5, 'medium_risk_1': 7, 'melanoma_9':2, 'melanoma_3':2, 'melanoma_27':5, 'melanoma_8':2}
    blur_iter={'medium_risk_10':15, 'melanoma_4':20, 'melanoma_3':15, 'melanoma_27':25, 'melanoma_8':15, 'melanoma_25':15, 'melanoma_17':25}
    choice=input("Do you want to process a single specific image ar a category? Reply with 'image' or 'category or 'exit' to quit: ")
    if(choice=='image'):
        
        name=input("Insert name of input picture to analyse: ")
        filein="./data/"+name+".jpg"
        
        #read image
        image = im.imread(filein)
        N1,N2, N3=image.shape
        plt.figure()
        plt.imshow(image)
        plt.title("Mole to analyse")
        plt.pause(0.1)
        
        #median_filter on image to blur
        if name in blur_iter.keys():
            image_filt=apply_blur(image, blur_iter[name])
        else:
            image_filt=apply_blur(image)
        plt.figure()
        plt.imshow(image_filt)
        plt.title("Mole blurred by median filter")
        plt.pause(0.1)
        
        #quantize image
        if name in n_clusters.keys():
            mole,centroids=quant_image(image_filt, n_clusters[name])
        else:
            mole, centroids=quant_image(image_filt)
        N1, N2=mole.shape
        mole_to_plot=np.zeros((N1,N2,3)).astype('uint8')
        #plot quantized mole
        for i in range(centroids.shape[0]):
            index=np.where(mole==i)
            mole_to_plot[index]=centroids[i,:]
        
        plt.figure()
        plt.imshow(mole_to_plot)
        plt.title("Quantized mole using N=%d clusters"%centroids.shape[0])
        plt.pause(0.1)
        
        mole_color=np.argmin(np.sum(centroids, axis=1))
        molebn=((mole!=mole_color).astype('uint8'))
        plt.figure()
        plt.imshow(molebn)
        plt.title("Extracted mole")
        plt.pause(0.1)
           
        
        #remove not mole pixel outside using DBSCAN
        X=mole_by_coord(molebn)
        pic_centre=np.array([N1/2, N2/2]).reshape(1,2)
        labels, mole_label=spatial_filtering(np.array(X), pic_centre)
        n=0
        for r in X:
            molebn[N1-1-r[0]][r[1]]=labels[n]
            n+=1
            
        molebn=((molebn!=mole_label).astype('uint8'))
        plt.figure()
        plt.imshow(molebn)
        plt.title("Mole without external noise")
        plt.pause(0.1)
        
        # reduce image around mole and filter the binary image for better results
        N1,N2=molebn.shape
        X=np.array(mole_by_coord(molebn))
        x_max=np.amax(X[:,0])
        x_min=np.amin(X[:,0])
        y_max=np.amax(X[:,1])
        y_min=np.amin(X[:,1])
        molebn=crop(molebn, x_max, x_min, y_max, y_min)
        #crop also original image to superimpose the 2 later on...
        image=crop(image,x_max, x_min, y_max, y_min).astype('uint8')
        
        plt.figure()
        plt.imshow(molebn)
        plt.title("Mole cropped with 20px border")
        plt.pause(0.1)
        
        #filter again the binary image to remove noise
        molebn=median_filter(molebn,3)
        molebn=median_filter(molebn,3)
        molebn=median_filter(molebn,3)
        
        #compute centre
        N1,N2=molebn.shape
        X=np.array(mole_by_coord(molebn))
        fig, ax = plt.subplots()
        ax.imshow(molebn)    
        mole_mean=mole_center(X)
        ax.plot(mole_mean[1], N1-1-mole_mean[0], 'ro')
        plt.title("Mole with its center")
        plt.pause(0.1)
        
        #edges
        X=mole_by_coord(molebn)
        mole_edges=sobel_filter(molebn)
        #again remove noise after edges extraction
        mole_edges=median_filter(mole_edges, 3)
        fig, ax = plt.subplots()
        ax.imshow(mole_edges)    
        ax.plot(mole_mean[1], N1-1-mole_mean[0], 'ro')
        plt.title("Mole edges extracted by sobel filter")
        plt.pause(0.1)
        
        #remove holes in the mole
        X=mole_by_coord(mole_edges)
        clean_mole_edges, perimeter=remove_holes(np.array(X), mole_edges)
        fig, ax = plt.subplots()
        ax.imshow(clean_mole_edges)    
        ax.plot(mole_mean[1], N1-1-mole_mean[0], 'ro')
        plt.title("Mole without holes")
        plt.pause(0.1)
        print("The mole perimeter is ", perimeter)
        
        #superimpose the original image and the found contour 
        X=np.array(mole_by_coord(clean_mole_edges))
        fig, ax = plt.subplots()
        ax.imshow(image)
        x=np.array(X[:,1])
        y=N1-1-np.array(X[:,0])
        plt.title("Original mole image with found border")
        ax.scatter(x,y,c='r', marker='o')
        plt.pause(0.1)
         
        #fill the mole
        full_mole=flood_fill(clean_mole_edges, [mole_mean[0], mole_mean[1]])
        
        #indentation
        ind=indentation(full_mole, perimeter)
        print("Mole has indentation coefficient: ", ind)
            
        #symmetry
        X=mole_by_coord(full_mole)
     
        m, sym =symmetry(X, mole_mean[1], mole_mean[0], name)
        
        print("Best symmetry is found using axis with slope", m)
        asym=1-sym
        print("Asymmetry value:", asym)
        N1,N2=full_mole.shape
        fig, ax = plt.subplots()
        ax.matshow(full_mole)    
        ax.plot(mole_mean[1], N1-1-mole_mean[0], 'ro')
        x=np.arange(0, N2, 1)
        if(m=='inf'):
            ax.axvline(x=mole_mean[1], color = 'b',  ls='-.')
        else:
            y=N1-1-(m*(x-mole_mean[1])+(mole_mean[0])) 
            y1=y[y<N1-1]
            y_to_plot=((y1[y1>0]))
            x_to_plot=(-(y_to_plot-N1+1+mole_mean[0])/m)+mole_mean[1]
            ax.plot(x_to_plot,y_to_plot, '-.b')
        plt.title("Mole with its main axis of symmetry")
        plt.pause(0.1)
        
    if(choice=='category'):
        
        categ=input("Insert category to analyse: ")
        asymmetry_vec=np.zeros((n_images[categ],1))
        indentation_vec=np.zeros((n_images[categ],1))
        results=open("./"+categ+"_results.txt", "w")
        results.write("Perimeter Indentation m of best symmetry axis asymmetry\n")
        for ID in range(n_images[categ]):    
            plt.close('all')
            ident=ID+1
            name=categ+'_'+str(ident)
            filein="./data/"+name+".jpg"
            
            #read image
            image = im.imread(filein)
            N1,N2, N3=image.shape
            
            
            #median_filter on image to blur
            if name in blur_iter.keys():
                image_filt=apply_blur(image, blur_iter[name])
            else:
                image_filt=apply_blur(image)
                plt.figure()
            plt.title("Mole blurred by median filter")
            plt.imshow(image_filt)
            plt.savefig("./fig/"+name+"_filt.jpg")
            plt.close()
            
            #quantize image     
            if name in n_clusters.keys():
                mole,centroids=quant_image(image_filt, n_clusters[name])
            else:
                mole, centroids=quant_image(image_filt)
            N1, N2=mole.shape
            mole_to_plot=np.zeros((N1,N2,3)).astype('uint8')
            #plot quantized mole
            for i in range(centroids.shape[0]):
                index=np.where(mole==i)
                mole_to_plot[index]=centroids[i,:]
            
            plt.figure()
            plt.imshow(mole_to_plot)
            plt.title("Quantized mole using N=%d clusters"%centroids.shape[0])
            plt.savefig("./fig/"+name+"_quant.jpg")
            plt.close()
            
            mole_color=np.argmin(np.sum(centroids, axis=1))
            molebn=((mole!=mole_color).astype('uint8'))
        
            plt.figure()
            plt.title("Extracted mole")
            plt.imshow(molebn)
            plt.savefig("./fig/"+name+"_bin.jpg")
            plt.close()
               
            
            #remove not mole pixel outside using DBSCAN
            X=mole_by_coord(molebn)
            pic_centre=np.array([N1/2, N2/2]).reshape(1,2)
            labels, mole_label=spatial_filtering(np.array(X), pic_centre)
            n=0
            for r in X:
                molebn[N1-1-r[0]][r[1]]=labels[n]
                n+=1
                
            molebn=((molebn!=mole_label).astype('uint8'))
            plt.figure()
            plt.imshow(molebn)
            plt.title("Mole without external noise")
            plt.savefig("./fig/"+name+"_clean.jpg")
            plt.close()
            
            # reduce image around mole and filter the binary image for better results
            N1,N2=molebn.shape
            X=np.array(mole_by_coord(molebn))
            x_max=np.amax(X[:,0])
            x_min=np.amin(X[:,0])
            y_max=np.amax(X[:,1])
            y_min=np.amin(X[:,1])
            molebn=crop(molebn, x_max, x_min, y_max, y_min)
            #crop also original image to superimpose the 2 later on...
            image=crop(image,x_max, x_min, y_max, y_min).astype('uint8')
           
            
            plt.figure()
            plt.imshow(molebn)
            plt.title("Mole cropped witj 20px border")
            plt.savefig("./fig/"+name+"_crop.jpg")
            plt.close()
            
            molebn=median_filter(molebn,3)
            molebn=median_filter(molebn,3)
            molebn=median_filter(molebn,3)
            
            #compute centre
            N1, N2=molebn.shape
            X=np.array(mole_by_coord(molebn))
            fig, ax = plt.subplots()
            ax.imshow(molebn)    
            mole_mean=mole_center(X)
            ax.plot(mole_mean[1], N1-1-mole_mean[0], 'ro')
            plt.title("Mole with its centre")
            plt.savefig("./fig/"+name+"_centre.jpg")
            plt.close()
            
            #edges
            mole_edges=sobel_filter(molebn)
            mole_edges=median_filter(mole_edges, 3)
            fig, ax = plt.subplots()
            ax.imshow(mole_edges)    
            ax.plot(mole_mean[1], N1-1-mole_mean[0], 'ro')
            plt.title("Mole edges extracted by sobel filter")
            plt.savefig("./fig/"+name+"_edges.jpg")
            plt.close()
            
            #remove holes in the mole
            X=mole_by_coord(mole_edges)
            clean_mole_edges, perimeter=remove_holes(np.array(X), mole_edges)
            fig, ax = plt.subplots()
            ax.imshow(clean_mole_edges)    
            ax.plot(mole_mean[1], N1-1-mole_mean[0], 'ro')
            plt.title("Mole without holes")
            plt.savefig("./fig/"+name+"_no_holes.jpg")
            plt.close()
            results.write(str(perimeter)+" ")
            
            #superimpose the original image and the found contour 
            X=np.array(mole_by_coord(clean_mole_edges))
            fig, ax = plt.subplots()
            ax.imshow(image)
            x=np.array(X[:,1])
            y=N1-1-np.array(X[:,0])
            plt.title("Original mole image with found border")
            ax.scatter(x,y,c='r', marker='o')
            plt.savefig("./fig/"+name+"_originaledge.jpg")
            plt.close()
            
            #fill the mole
            full_mole=flood_fill(clean_mole_edges, [mole_mean[0], mole_mean[1]])
            
            #indentation
            ind=indentation(full_mole, perimeter)
            indentation_vec[ID]=ind
            results.write(str(round(ind,3))+" ")
            
            
            #symmetry
            X=mole_by_coord(full_mole)
            
            m, sym =symmetry(X, mole_mean[1], mole_mean[0], name)
            results.write(str(m)+" ")
            asym=1-sym
            results.write(str(round(asym,3))+"\n")
            asymmetry_vec[ID]=asym
            N1,N2=full_mole.shape
            fig, ax = plt.subplots()
            plt.title("Mole with its main axis of symmetry")
            ax.matshow(full_mole)    
            ax.plot(mole_mean[1], N1-1-mole_mean[0], 'ro')
            x=np.arange(0, N2, 1)
            if(m=='inf'):
                ax.axvline(x=mole_mean[1], color = 'b',  ls='-.')
            else:
                y=N1-1-(m*(x-mole_mean[1])+(mole_mean[0])) 
                y1=y[y<N1-1]
                y_to_plot=((y1[y1>0]))
                x_to_plot=(-(y_to_plot-N1+1+mole_mean[0])/m)+mole_mean[1]
                ax.plot(x_to_plot,y_to_plot, '-.b')
            plt.savefig("./fig/"+name+"_final.jpg")
            plt.close()
            
        #category statistics
        mean_asymmetry=round(asymmetry_vec.mean(),3)
        mean_indentation=round(indentation_vec.mean(),3)
        
        std_asymmetry=round(asymmetry_vec.std(),3)
        std_indentation=round(indentation_vec.std(),3)
        
        results.write("For category {}: \nmean indentation={} mean asymmetry={} \n indentation std={} asymmetry std={}".format(categ, mean_indentation, mean_asymmetry, std_indentation, std_asymmetry))
        results.close()
        #indentation histogram
        plt.figure()
        plt.hist(np.around(indentation_vec,3),bins=5)
        plt.xlabel('indentation')
        plt.ylabel('Absolute frequency of indentation value')
        plt.title("Histogram of indentation for category: "+categ)
        plt.grid()
        plt.savefig("./fig/Histogram_indet_"+categ+".jpg")
        plt.close()
        
        
        #symmetry histogram
        plt.figure()
        plt.hist(np.around(asymmetry_vec,3),bins=5)
        plt.xlabel('Asymmetry')
        plt.grid()
        plt.ylabel('Absolute frequency of asymmetry value')
        plt.title("Histogram of asymmetry for category: "+categ)
        plt.savefig("./fig/Histogram_asym_"+categ+".jpg")
        plt.close()