
# Building Swin Transformer from Scratch using PyTorch: Hierarchical Vision Transformer using Shifted Windows

[

![Mishra](https://miro.medium.com/v2/da:true/resize:fill:88:88/0*-Sx-jbFtmQSk80p5)


# What are Swin Transformers

The Vision Transformers leveraged the window-based self-attention (computing attention score within every window) which lacks connection across the windows themselves. Which limits its modeling power. Identifying this issue Swin Transformer proposed the idea of Shifted Windows to introduce cross-window connections or to enable communication within the windows.

Another reason for introducing the Shifted window is that the standard transformer conducts global self-attention across all the tokens. Which is computationally expensive where using Shifted Windows the complexity decreases from quadratic to linear.

The Swin Transformer is divided into three following things:

1. Patch Embedding
2. Patch Merging
3. Shifted Window
4. Positional Embedding
5. Window Mask

The architecture is similar to that of the standard ViT. With the addition of Patch Merging, Shifted Windows, and Masking. Due to the induction of Shifted Windows, there’s a slight change in how one would calculate positional embedding.

![](https://miro.medium.com/v2/resize:fit:770/1*njVvNcQ583BRMEs4jLdzlw.png)

Figure 1

## Patch Embedding

This is the same as before, you take an input Image shape (B, C, H, W), pass it down a conv2d, and rearrange it to (B, T(number of patches), Embed_dim)

class SwinEmbedding(nn.Module):  
    def __init__(self, patch_size=4, emb_size=96):  
        super().__init__()  
        self.linear_embedding = nn.Conv2d(3, emb_size, kernel_size = patch_size, stride = patch_size)  
        self.rearrange = Rearrange('b c h w -> b (h w) c')  
          
    def forward(self, x):  
        x = self.linear_embedding(x)  
        x = self.rearrange(x)  
        return x

## Patch Merging

In the figure above we see the input (B, T, 48) where T: (H/patch, W/patch) and 48 is just the linear (patch X patch X 3), we let the SwinEmbedding take care of all that and make it (B, T, C). In Patch Merging, we would want to reduce the token length by 4 and increase the embedding dim by 2.

> i.e. B, T(H/4 X W/4), C — > B, T(H/8 X W/8), 2C

class PatchMerging(nn.Module):  
    def __init__(self, emb_size):  
        super().__init__()  
        self.linear = nn.Linear(4*emb_size, 2*emb_size)  
  
    def forward(self, x):  
        B, L, C = x.shape  
        H = W = int(np.sqrt(L)/2)  
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s1 s2 c)', s1=2, s2=2, h=H, w=W)  
        x = self.linear(x)  
        return x

Here we’ve added a linear layer at the end because during rearranging the embedding dimension increases by 4X but we want it to make 2X the previous dim.

The Patch merging takes significant participation in creating the hierarchical representation of the input. That’s one of the main features proposed, reducing the spatial dimension of the input feature map as the depth increases so to add complexity to the ViT models.

![](https://miro.medium.com/v2/resize:fit:770/1*eccqwwRCGArf6CfjVT01cA.png)

# Shifted Window Attention

It’s again the same as the basic self-attention, with the addition of shifted windows and masking.

Let’s start with understanding the shifted windows:

![](https://miro.medium.com/v2/resize:fit:740/1*fGr3gsejo4p-SoiI24e5YQ.png)

Here we are taking the window size = 4 X 4 in layer 1. Each window will go through a self-attention module as in the traditional ViT. This part is known as _Window self-attention_. The next part i.e. layer 2 involves shifting the window by (window_size/2, window_size/2) this part is known as _Shifted Window Self Attention_.

![](https://miro.medium.com/v2/resize:fit:770/1*ENM9cBry4dpZLrzyJRWRpQ.jpeg)

In layer 2 we just shifted the four windows in layer 1 to the right (by window_size(4)/2 = 2) and down (by window_size(4)/2 = 2). Now after the window shift, we are left with some extra dummy space at the right and bottom of the image. Thus we simply replace the extra space with A at the bottom right corner, C at the bottom, and B at the right. This is the main proposed idea of the paper we wanted to somewhat enable communication between the windows or in other words “cross window” attention.

  
class ShiftedWindowMSA(nn.Module):  
    def __init__(self, emb_size, num_heads, window_size=7, shifted=True):  
        super().__init__()  
        self.emb_size = emb_size  
        self.num_heads = num_heads  
        self.window_size = window_size  
        self.shifted = shifted  
        self.linear1 = nn.Linear(emb_size, 3*emb_size)  
  
    def forward(self, x):  
        h_dim = self.emb_size / self.num_heads  
        height = width = int(np.sqrt(x.shape[1]))  
        x = self.linear1(x)  
          
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)  
          
        if self.shifted:  
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))  
          
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)         
  
        Q, K, V = x.chunk(3, dim=6)  
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)  
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)  

Here we use a very standard approach to calculate attention score (wei).

We first pass the input (B, T, C) to a linear layer, obtain 3*emb_size, and later split it into dim c and k. (eg: linear(B, T, 10) → B, T, 30 → B, T, 10, 3). Also, one thing to notice here is we rearrange the token length ( B, T (h w) (c k) -> B, h, w, c, k ) we do this because we will be calculating attention within a window, a window is nothing but a partition (generally square as shown in layer 1) which contains tokens of shape (window size, window size).

As mentioned there are two types of attention, window and shited window. if we are using a shifted window, we would want to shift them as explained previously, thus we use torch.roll to perform this task.

After our input is of shape(B, H, W, C, K) where H is the height of the token feature map and W is the width of the token feature map. we now create windows of some window_size as shown in the layer 1 image. For ex: Here let's say H and W are 32 X 32 and the window size is 4 then it will create a total of 32/4 X 32/4 windows with each window having 4 X 4 (16) tokens.

_i.e. (B (32 w1) (32 w2) (e H) k → B H 8 8 (16) e k ) where e is new embedding and H is the number of heads ( e*H = C)_

Note: Do not confuse H and W to be the height and width of the image, we have already passed the image through embedding to obtain input (B, T(H X W), C) now the H X W obtain are height and width of tokens each containing embedding vectors C.

## Window Masking

In layer 1 + 1 as explained above, after shifting windows, we replace the new space with A, B, and C. But there is a problem here we do not want the A, B, and C to communicate with their surroundings entirely, as they are not a part of it.

![](https://miro.medium.com/v2/resize:fit:538/1*RbxdXt8j2766LEsWY9z_nQ.png)

This is the newly shifted window where we gonna perform attention for each window. Let’s take the bottom right window (A 1,2,3 and 4) for example. Here if we perform self-attention, all four blocks would be considered related to each other. But that is not true, all four attentions are completely not related to each other as they have features from the other side of the image. It would not be a good idea to perform Attention to all of them as they are not neighbors but rather rearranged.

Let's add the mask to our Shiftend Window attention.

class ShiftedWindowMSA(nn.Module):  
    def __init__(self, emb_size, num_heads, window_size=7, shifted=True):  
        super().__init__()  
        self.emb_size = emb_size  
        self.num_heads = num_heads  
        self.window_size = window_size  
        self.shifted = shifted  
        self.linear1 = nn.Linear(emb_size, 3*emb_size)  
    def forward(self, x):  
        h_dim = self.emb_size / self.num_heads  
        height = width = int(np.sqrt(x.shape[1]))  
        x = self.linear1(x)  
          
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)  
          
        if self.shifted:  
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))  
          
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)              
          
        Q, K, V = x.chunk(3, dim=6)  
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)  
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)  
          
        if self.shifted:  
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()  
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')  
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')  
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)  
            wei[:, :, -1, :] += row_mask  
            wei[:, :, :, -1] += column_mask  
          
        wei = F.softmax(wei, dim=-1) @ V   

I’m not gonna go very deep into the masking code, but it’s really important to understand what we are doing and why we are doing it. I strongly recommend you to watch this quick video explanation [https://www.youtube.com/watch?v=s0yiRi_pr10&list=PL9iXGo3xD8jokWaLB8ZHUkjjv5Y_vPQnZ&index=5](https://www.youtube.com/watch?v=s0yiRi_pr10&list=PL9iXGo3xD8jokWaLB8ZHUkjjv5Y_vPQnZ&index=5).

This will explain to you why are we doing this and how the masking code works as well.

## Positional Embedding

The Swin model uses a relative positional embedding.

![](https://miro.medium.com/v2/resize:fit:770/1*8Z19hHAW1FtuwEdVSDU6dQ.png)

Figure

These are the following steps to create relative bias positional embeddings

1. In the Figure above we are assuming the window size to be 3.
2. Create an indices matrix along the x-axis and y-axis as shown in the image above.
3. Here for window size 3, we create a matrix of shape (3² X 3²). We see in the figure above that in the first row in x-axis matrics (1 2 3……9) we have 1 2 3 marked as 0 which is their distance from the first row and 4 5 6 is marked with -1 ( as they are one row below the first row) following the same logic we create the rest of the matrics
4. The same is done for the y-axis matrix but instead, we do it alone the y-axis. Thus now for the first row in the y-axis, we see 2 is 1 column right of it so we put -1, and then 3 is two columns right of the first row so we put -2 at its place in the y-axis matrix following the same logic we fill the entire matrix.
5. Finally, we add (M — 1) to every element in both the x and y-axis matrix where M is the window size, in our example 3. We do this to make sure everything is in the range [0, 2M — 1] because, for a window size 3, the relative positions would have to be between [-2, +2]. Then we multiply the x-axis matrix by (2M -1) and add it to the y-axis matrix. The final **Relative Position index** matrix will have a range of (M² — 1)X(M² — 1) which is (0, 25) here.

![](https://miro.medium.com/v2/resize:fit:770/1*wiCWnSMsnanyadQFv8yduw.png)

Here we should be recalling what are we doing and whom are we doing it for. We have a tensor x (shape: B, H, Wh, Ww, win_size², win_size²) and we want to create positional embedding matrics for it, of shape (win_size², win_size²). What we have is a Relative Position Index of shape (M, M) where M is window_size.

In a window size M of 3, we have the position range of (0, 5), and for creating position embedding for matrics shaped (M², M²) we would need the range of (0, 25) as explained in the 5th point.

In the Relative Position Index of shape (M, M) we have 25 (for M=3) unique indices thus it would be better to have the same weight of an Index (say 12) everywhere an index is 12. That is why we pass it to a matrix of shape 5X5 (few implementations use a matrix of shape 5X5 or a linear 25, doesn’t matter and gives the same result) and get the parameters for the Final Positional Embedding.

class ShiftedWindowMSA(nn.Module):  
    def __init__(self, emb_size, num_heads, window_size=7, shifted=True):  
        super().__init__()  
        self.emb_size = emb_size  
        self.num_heads = num_heads  
        self.window_size = window_size  
        self.shifted = shifted  
        self.linear1 = nn.Linear(emb_size, 3*emb_size)  
        self.linear2 = nn.Linear(emb_size, emb_size)  
  
        self.pos_embeddings = nn.Parameter(torch.randn(window_size*2 - 1, window_size*2 - 1))  
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))  
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]  
        self.relative_indices += self.window_size - 1  
  
    def forward(self, x):  
        h_dim = self.emb_size / self.num_heads  
        height = width = int(np.sqrt(x.shape[1]))  
        x = self.linear1(x)  
          
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)  
          
        if self.shifted:  
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))  
          
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)              
          
        Q, K, V = x.chunk(3, dim=6)  
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)  
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)  
          
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]  
        wei += rel_pos_embedding  
          
        if self.shifted:  
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()  
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')  
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')  
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)  
            wei[:, :, -1, :] += row_mask  
            wei[:, :, :, -1] += column_mask  
          
        wei = F.softmax(wei, dim=-1) @ V  
          
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)  
        x = rearrange(x, 'b h w c -> b (h w) c')  
          
        return self.linear2(x)

The only addition to the code is the four lines for positional embedding. Trust me it does exactly what I explained earlier. The official Implementation from Microsoft is a bit different, to know more about the code see this YouTube tutorial: [https://www.youtube.com/watch?v=iTHK0FDWJys&list=PL9iXGo3xD8jokWaLB8ZHUkjjv5Y_vPQnZ&index=8](https://www.youtube.com/watch?v=iTHK0FDWJys&list=PL9iXGo3xD8jokWaLB8ZHUkjjv5Y_vPQnZ&index=8), This should clear if you have any other doubts regarding position embeddings.

We rearrange the wei back to its initial shape (B, H, W, C) where H is the height of the tokens, W is the width of the tokens and C is the embedding dimension.

Lastly, we rearrange x to its original dimension (B, T, C)

# Swin Encoder

![](https://miro.medium.com/v2/resize:fit:557/1*-AQRAtXXuXPVuXZpEQ2Uxw.png)

We are at the final section of this article, This is a single Swin Encoder that involves the standard ViT architecture’s encoder module, one with normal window-based attention passing its output as an input to the shifted window attention encoder.

class MLP(nn.Module):  
    def __init__(self, emb_size):  
        super().__init__()  
        self.ff = nn.Sequential(  
                         nn.Linear(emb_size, 4*emb_size),  
                         nn.GELU(),  
                         nn.Linear(4*emb_size, emb_size),  
                  )  
      
    def forward(self, x):  
        return self.ff(x)  
      
class SwinEncoder(nn.Module):  
    def __init__(self, emb_size, num_heads, window_size=7):  
        super().__init__()  
        self.WMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=False)  
        self.SWMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=True)  
        self.ln = nn.LayerNorm(emb_size)  
        self.MLP = MLP(emb_size)  
          
    def forward(self, x):  
        # Window Attention  
        x = x + self.WMSA(self.ln(x))  
        x = x + self.MLP(self.ln(x))  
        # shifted Window Attention  
        x = x + self.SWMSA(self.ln(x))  
        x = x + self.MLP(self.ln(x))  
          
        return x

# Putting It All Together

Finally, we are gonna put every module together to build the SwinTransformer class.

class Swin(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.Embedding = SwinEmbedding()  
        self.PatchMerging = nn.ModuleList()  
        emb_size = 96  
        num_class = 5  
        for i in range(3):  
            self.PatchMerging.append(PatchMerging(emb_size))  
            emb_size *= 2  
          
        self.stage1 = SwinEncoder(96, 3)  
        self.stage2 = SwinEncoder(192, 6)  
        self.stage3 = nn.ModuleList([SwinEncoder(384, 12),  
                                     SwinEncoder(384, 12),  
                                     SwinEncoder(384, 12)   
                                    ])  
        self.stage4 = SwinEncoder(768, 24)  
          
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size = 1)  
        self.avg_pool_layer = nn.AvgPool1d(kernel_size=49)  
          
        self.layer = nn.Linear(768, num_class)  
  
    def forward(self, x):  
        x = self.Embedding(x)  
        x = self.stage1(x)  
        x = self.PatchMerging[0](x)  
        x = self.stage2(x)  
        x = self.PatchMerging[1](x)  
        for stage in self.stage3:  
            x = stage(x)  
        x = self.PatchMerging[2](x)  
        x = self.stage4(x)  
        x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))  
        return x

The code is pretty self-explanatory, we start with creating swin embedding for our Image. For example let's consider an Image of Shape (B, C, H, W) and pass it to the Swin Embedding we get (B, T(H/4 X W/4), C).

We create 4 different stages as proposed (see Figure 1) in the architecture. As explained in the patch merging section we create a list of PatchMerging modules for different embeddings and use them successively on the corresponding stage’s output.

The output after the stage1 is (B, T, C) passing it to PactchMerging we get x shape (B, T/4, 2C), after stage2 and PatchMerging (B, T/16, 4C), after stage3 and PatchMerging (B, T/64, 8C) and after stage4 we have input x of shape (B, T/64, 8C) where T is (H/4 X W/4).

Finally, we fit the model. That is it. There we have our Swin Transformer.

if __name__ == '__main__':  
    # Usage Example (assuming num_classes = 5)  
  
    x = torch.rand(1, 3, 224, 224)  
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    x = x.type(torch.FloatTensor).to(device)  
    model = Swin().to(device)

If you liked my work, please consider giving it a clap or follow. You can find the entire code here on my GitHub repository: [https://github.com/mishra-18/ML-Models/blob/main/Vission%20Transformers/swin_transformer.py](https://github.com/mishra-18/ML-Models/blob/main/Vission%20Transformers/swin_transformer.py)

Thanks for reading..