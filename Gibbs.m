% Function to implement Gibbs sampling. See "Probabilistic Topic Models" by
% Mark Steyvers and Tom Griffiths

function [C_WT, C_DT, vocab, topicRep] = Gibbs(K, w, d)

    % K = number of topics
    % w = vector of words
    % d = vector of documents associated with each word
    % z = vector of topics associated with each word (initialized to a random topic)

    N = numel(w);                       % total number of words in the corpus
    vocab = unique(w,'stable');         % set of unique words in combined training + test data set
    V = numel(vocab);                   % number of unique words in vocab
    D = max(d);                         % number of documents
    
    % INITIALIZATIONS
    
    numIterations = 100;                % recommended by PP4 worksheet
    alpha = 0.1;                        % hyperparameter for theta
    beta = 0.01;                        % hyperparamter for beta
    
    z = floor(rand(N,1) * K) + 1;       % initialize to random topics
    %z = ones(N,1);
    
    P = zeros(K,1);                     % probability of assigning word(i) to each topic
    idxPi = randperm(N);                % create a random index to order words
      
    % INITIALIZE PHI AND THETA
    
    C_WT = zeros(K,V);                  % PHI, [numTopics x numWordsVocab], contains word counts
    C_DT = zeros(D,K);                  % THETA, [numDocuments x numTopics], contains topic counts 
    
    idxWord = PP1ReturnIndexK(w,vocab); % index each of the N words with a vocabulary word
            
    for i = 1:N                         % init the two matrices 
        
        word = idxWord(idxPi(i));
        topic = z(idxPi(i));
        doc = d(idxPi(i));
        
        C_WT(topic,word) = C_WT(topic,word) + 1;
        C_DT(doc,topic) = C_DT(doc,topic) + 1;
        
    end
    
    %%%%%%%%%%%%%%%% START MAIN ALGORITHM %%%%%%%%%%%%%%
    
    for iter = 1:numIterations
                
        for n = 1:N
            
            word = idxWord(idxPi(n));
            topic = z(idxPi(n));
            doc = d(idxPi(n));
            
            C_WT(topic,word) = C_WT(topic,word) - 1;            % decrement PHI
            C_DT(doc,topic) = C_DT(doc,topic) - 1;              % decrement THETA
           
            for k = 1:K
                
               PHI = (C_WT(k,word) + beta) / (sum(C_WT(k,:)) + V*beta);
               THETA = (C_DT(doc,k) + alpha) / (sum(C_DT(doc,:)) + K*alpha);
               P(k) = PHI * THETA;
                
            end
                        
            P = P/sum(P);                                       % normalize P(k)     
            topic = find((rand <= cumsum(P)),1,'first');        % sample a topic from P(k)
            z(idxPi(n)) = topic;
                        
            C_WT(topic,word)  = C_WT(topic,word) + 1;           % increment PHI
            C_DT(doc,topic) = C_DT(doc,topic) + 1;              % increment THETA
            
        end % for n
                
    end % for iter 
    
    topicRep = zeros(D,K);                                        % init topic representation
    for doc = 1:D
        topicRep(doc,:) = (C_DT(doc,:) + .1 ) / ( sum(C_DT(doc,:),2) + K*.1 );  % topic representation
    end

    
end % function