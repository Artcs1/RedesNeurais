import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import operator as op

def sigmoid(net): # função sigmoide
    return (1/(1+np.exp(-net)))

def derivadanet(net):
    return (net*(1-net))

def feedforward(X,model,function = sigmoid):
    
    fnetH = []
    Whidden = model[3]
    Woutput = model[4]
    
    X = np.concatenate((X,np.ones(1)),axis = 0)
    
    for i in range (len(model[1])):
        netH = np.dot(Whidden[i],X)
        fnetH.append(function(netH))
        X = np.concatenate((fnetH[i],np.ones(1)),axis = 0)
        
    netO = np.dot(Woutput,X)
    fnetO = function(netO)
    
    
    return [fnetO,fnetH]

def backpropagation(model,X,Y,eta = 0.1,momentum = 0.2 ,threshold = 1e-7, dnet = derivadanet,maxiter = 80):
    sError = 2*threshold
    c = 0
    while sError > threshold and c < maxiter: # criterio de parada
        sError = 0
        
        tam = Y.shape[0]
        
        
        for i in np.arange(tam):
        
            Xi = X[i,]
            Yi = Y[i,]
        
            results = feedforward(Xi,model);

            O = results[0]
            #Error
            Error = Yi-O
        
            sError = sError + np.sum(np.power(Error,2))
        
            #Treinamento capa de salida
            h = len(model[1])
            
            Hp = np.concatenate((results[1][h-1],np.ones(1)),axis=0)
            Hp = np.reshape(Hp,(model[1][h-1]+1,1))
            
            deltaO = Error * dnet(results[0])
            dE2_dw_O = np.dot((-2*deltaO.reshape((deltaO.shape[0],1))),np.transpose(Hp)) 
            Voutput = np.zeros(dE2_dw_O.shape)
            Voutput = momentum*Voutput + dE2_dw_O
            #Treinamento capa intermedia
        
            deltaH = []
            dE2_dw_h = []
            Vhidden = []
            
            delta = deltaO
            Wk = model[4][:,0:model[1][h-1]]
        
            for i in range(h):
                deltaH.append(0)
                dE2_dw_h.append(0)
                Vhidden.append(0)
            
            for h in range(len(model[1])-1,0,-1):
                Xp = np.concatenate((results[1][h-1],np.ones(1)),axis = 0 )
                Xp = np.reshape(Xp,(1,model[1][h-1]+1))
                
                deltaH[h] = np.dot(delta,Wk) 
                dE2_dw_h[h] = deltaH[h].reshape((deltaH[h].shape[0],1)) * (np.dot(-2*dnet(results[1][h]).reshape((results[1][h].shape[0],1)),Xp))
                Vhidden[h] = momentum*Vhidden[h] + dE2_dw_h[h]
                
                
                delta = deltaH[h]
                Wk = model[3][h][:,0:model[1][h-1]]
            
            Xp = np.concatenate((Xi,np.ones(1)),axis=0)
            Xp = np.reshape(Xp,(1,model[0]+1))
            
            deltaH[0] = (np.dot(delta,Wk))
            dE2_dw_h[0] = deltaH[0].reshape((deltaH[0].shape[0],1)) * (np.dot(-2*dnet(results[1][0]).reshape((results[1][0].shape[0],1)),Xp))
            Vhidden[0] =momentum*Vhidden[0] + dE2_dw_h[0]
            #atualização dos pesos
        
            model[4] =  model[4] -  eta*Voutput            
            for i in range(len(model[1])):
                model[3][i] =  model[3][i] -  eta*Vhidden[i] 
        
        #contador
        
        sError = sError / tam
        c = c+1
        #print("iteração ",c)
        #print("Error:",sError)
        #print("\n");
    

    return model
def mlp(Isize = 10,Hsize = [2,4] ,Osize = 3):
    
    # Isize tamano da camada de entrada
    # Osize tamano da camada de salida
    # Hsize tamano de camada oculta

    Whidden = []
    Vmomentum = []
    previous_length = Isize    
    for i in range (len(Hsize)):
        Whidden.append(np.random.random_sample((Hsize[i],previous_length +1)) - 0.5 )
        Vmomentum.append(np.zeros((Hsize[i],previous_length +1)));
        previous_length = Hsize[i]    

    Woutput = np.random.random_sample((Osize,previous_length +1)) - 0.5     
    model = [Isize,Hsize,Osize,Whidden,Woutput,Vmomentum]
    
    return model

def normalizacion(Data):
    
    for i in  np.arange(Data.shape[1]): 
        Data[:,i] = ( Data[:,i] - np.min(Data[:,i]) )/( np.max(Data[:,i]) - np.min(Data[:,i]) )
    return Data

def binarizar(Y,siz = 3):
    Y2 = np.zeros((Y.shape[0],siz))
    for i in np.arange(Y.shape[0]):
        Y2[i,int(Y[i])-1] = 1
        
    return Y2

def clasificacion(model,X,Y):
    acierto = 0;
    tam = Y.shape[0]
    for i in np.arange(tam):
        Yesperado = feedforward(X[i,],model)[0];
        Yi = np.round(Yesperado)
        if np.sum(Yi - Y[i,]) == 0: 
             acierto = acierto +1
    return (acierto*100)/tam


def regresion(model,X,Y):
    serror = 0;
    tam = Y.shape[0]
    for i in np.arange(tam):
        Yi = feedforward(X[i,],model)[0];
        serror = serror + np.power(np.sum(Yi - Y[i,]),2)
    return serror/tam

def sub_sampler(data_samples, ratio, verbose=False):
    top_id = int(data_samples.shape[0] * ratio)
        
    rows_id = np.arange(data_samples.shape[0])
    sub_sample_ids_train = np.random.choice(rows_id, top_id, replace=False)
        
    sub_sample_ids_test = np.setdiff1d(rows_id, sub_sample_ids_train)
    
    return [sub_sample_ids_train,sub_sample_ids_test]

def wine_test(Data,siz = 0.7,maxiter = 100, eta = 0.7,momentum = 0):
 
    length = Data.shape[1]
    X = Data[:,1:length]
    Y = Data[:,0]

    X = normalizacion(X)
    Y = binarizar(Y, siz = 3)
    
    sizes = sub_sampler(X, siz)
    
    X1 = X[sizes[0],:]
    Y1 = Y[sizes[0],:]
    
    X2 = X[sizes[1],:]
    Y2 = Y[sizes[1],:]
    
    M = mlp(Isize = 13, Hsize = [2], Osize = 3)
    trained = backpropagation(M,X1,Y1,eta = eta , momentum = momentum , maxiter = maxiter)
    
    A = clasificacion(trained,X1,Y1)
    B = clasificacion(trained,X2,Y2)
    
    print("Particionamiento: "+ str(siz) +" Max.Iter: " + str(maxiter) + " Eta: " + str(eta) +  "Momentum:" + str(momentum))    
    print("Error na data de treinamento",A)
    print("Error na data de test",B)
    print("\n")
            
    return [A,B]

def music_test(Data,siz = 0.7, maxiter = 30,eta = 0.55,momentum = 0):
    
    Data = normalizacion(Data)
    
    length = Data.shape[1]
 
    X = Data[:,0:length-2]
    Y = Data[:,length-2:length]

    sizes = sub_sampler(X, siz)
    
    X1 = X[sizes[0],:]
    Y1 = Y[sizes[0],:]
    
    X2 = X[sizes[1],:]
    Y2 = Y[sizes[1],:]
   
    M = mlp(Isize = 68, Hsize = [2], Osize = 2)
    trained = backpropagation(M,X1,Y1,eta = eta , momentum = momentum , maxiter = maxiter)
    
    
    A= regresion(trained,X1,Y1)
    B= regresion(trained,X2,Y2)
    
    print("Particionamiento: "+ str(siz) +" Max.Iter: " + str(maxiter) + " Eta: " + str(eta))    
    print("Error na data de treinamento",A)
    print("Error na data de test",B)
    print("\n")
    
    
    return [A,B]

def main():
    
   datos = pd.read_csv("default_features_1059_tracks.data",sep = ",") # leitura dos dados
   MusicData = datos.values
   music_test(MusicData)
   
   datos2 = pd.read_csv("wine.data",sep = ",")
   WineData = datos2.values
   wine_test(WineData)
   
if __name__ == "__main__":
    main()
