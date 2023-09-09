# Personal-Movie-Recommendation

This project uses the Expectation-Maximization algorithm to build a simple movie recommendation system. The text files ${\it movies.txt}$, ${\it ids.txt}$, and ${\it ratings.txt}$ contain all the data. The last of these files contains a matrix of zeros, ones, and missing elements denoted by question marks. The $\langle i,j\rangle^{\rm th}$ element in this matrix contains the $i^{\rm th}$ user's rating of the $j^{\rm th}$ movie, according to the following key:

$$\begin{aligned}
1 \quad & recommended, \\
0 \quad & not\ recommend, \\
? \quad & not\ seen.
\end{aligned}$$

## Likelihood

Next step is to learn a naive Bayes model of these movie ratings, represented by the belief network shown below, with hidden variable $Z\in\\{1,2,\ldots,k\\}$ and partially observed binary variables $R_1,R_2,\ldots,R_{76}$ (corresponding to movie ratings).

<div align="center">
  
![image](https://github.com/aayushg97/Personal-Movie-Recommendation/assets/30308551/36f6f1f3-53ae-42a8-abb4-aed8686c2fe9)
</div>

This model assumes that there are $k$ different types of movie-goers, and that the $i^{\rm th}$ type of movie-goer, who represents a fraction $P(Z = i)$ of the overall population, likes the $j^{\rm th}$ movie with conditional probability $P(R_j = 1|Z = i)$. Let $\Omega_t$ denote the set of movies seen (and hence rated) by the $t^{\rm th}$ user. Then the likelihood of the $t^{\rm th}$ user's ratings is given by:

$$ P \left( \left(R_j = r_j^{(t)}\right)_ {j \in \Omega_t}\right)\ =\ \sum_{i=1}^k P(Z = i)\ \prod_{j\in\Omega_t} P\left(\left.R_j = r^{(t)}_j\right|Z = i\right).$$

## Expectation step

The Expectation step of this model is to compute, for each user, the posterior probability that they correspond to a particular type of movie-goer. Mathematically,

$$ P \left( Z=i \left| \left(R_j= r_j^{(t)}\right)_ {j \in \Omega_t}\right.\right)\ =\ \frac{P(Z= i)\ \prod_{j\in\Omega_t} P \left( \left. R_j=r^{(t)}_ j \right| Z=i \right)}{{\sum_{i'=1}}^k \ P(Z= i')\ \prod_{j\in\Omega_t} P\left(\left.R_j= r^{(t)}_j\right|Z=i'\right)}.$$

## Maximization step

The maximization step of the model is to re-estimate the probabilities $P(Z = i)$ and $P(R_j = 1|Z = i)$ that define the CPTs of the belief network. As a shorthand, let 

$$\rho_{it}\ =\ P\left(Z=i\left| \left(R_j= r_j^{(t)}\right)_{j\in\Omega_t}\right.\right)$$

denote the probabilities computed in the expectation step of the algorithm. Also, let $T$ denote the number of students. Then the EM updates are given by:

$$\begin{eqnarray*}
P(Z= i) & \leftarrow & \frac{1}{T} \sum_{t=1}^{T} \rho_{it}, \\ \\
P(R_j= 1|Z=i) & \leftarrow & 
  \frac{\sum_{ \\{ t | j \in \Omega_t \\} } \rho_{it}\ I \left(r^{(t)}_ {j}, 1 \right) + \sum_{ \\{ t | j \not \in \Omega_t \\} } \rho_{it}\ P(R_j=1|Z=i)}{{\sum_{t=1}}^{T}\ \rho_{it}}.
\end{eqnarray*}$$

## Implementation

The files <b>probZ_init.txt</b> and <b>probR_init.txt</b>, are used to initialize the probabilities $P(Z=i)$ and $P(R_j=1|Z=i)$ for a model with $k=4$ types of movie-goers. The code runs 256 iterations of the EM algorithm, computing the (normalized) log-likelihood

$${\cal L}\ =\ \frac{1}{T}\sum_{t=1}^T \log P\left(\left(R_j=r_j^{(t)}\right)_{j\in\Omega_t}\right)$$

at each iteration. 

<div align="center">

| Iteration | Log-Likelihood ${\cal L}$ |
| --- | --- |
| 0 |  -28.63 |
| 1 |  -19.35 |
| 2 |  -17.91|
| 4 | -17.08|
| 8 | -16.63|
| 16 | -16.29| 
| 32 | -15.80|
| 64 | -15.75|
| 128 | -15.74|
| 256 | -15.73|

</div>

## Personal movie recommendations

Any user ID in ${\it ids.txt}$ can be picked to determine the row of the ratings matrix that stores the users personal data. The code will compute the posterior probability in the expectation step, for this row from the trained model, and then compute the user's ${\it expected}$ ratings on the movies they haven't yet seen:

$$ P \left( R_l = 1 \left| \left( R_j = r_j^{(t)} \right)_ {j \in \Omega_t } \right. \right) \ =\ \sum_{i=1}^k P \left( Z=i \left| \left( R_j = r_j^{(t)} \right)_{j \in \Omega_t} \right. \right) P(R_l = 1 | Z = i)$$

for $l \not \in \Omega_t$

The list of these (unseen) movies, sorted by their expected ratings, is printed below.

<div align="center">
  
| name | score |
| --- | --- |
| Parasite | 0.868957 |
| La_La_Land | 0.793830 |
| Frozen | 0.760267 |
| Ready_Player_One | 0.757193 |
| Django_Unchained | 0.752938 |
| 21_Jump_Street | 0.751170 |
| Midnight_in_Paris | 0.725048 |
| The_Great_Gatsby | 0.715400 |
| Black_Swan | 0.692148 |
| Room | 0.684863 |
| Hidden_Figures | 0.676849 |
| Darkest_Hour | 0.673471 |
| 12_Years_a_Slave | 0.668835 |
| Pokemon_Detective_Pikachu | 0.644331 |
| Les_Miserables | 0.625264 |
| Phantom_Thread | 0.603146 |
| Pitch_Perfect | 0.598900 |
| Three_Billboards_Outside_Ebbing | 0.590423 |
| Rocketman | 0.579052 |
| Us | 0.572354 |
| The_Help | 0.571179 |
| Manchester_by_the_Sea | 0.548702 |
| Mad_Max:_Fury_Road | 0.548465 |
| Once_Upon_a_Time_in_Hollywood | 0.540396 |
| Dunkirk | 0.539244 |
| Her | 0.529423 |
| The_Girls_with_the_Dragon_Tattoo | 0.493689 |
| Terminator:_Dark_Fate | 0.475126 |
| The_Shape_of_Water | 0.461854 |
| The_Last_Airbender | 0.457509 |
| Drive | 0.439722 |
| Fifty_Shades_of_Grey | 0.434919 |
| Magic_Mike | 0.425444 |
| The_Farewell | 0.416342 |
| American_Hustle | 0.375991 |
| Good_Boys | 0.372371 |
| Bridemaids | 0.356023 |
| The_Hateful_Eight | 0.320158 |
| Hustlers | 0.293004 |
| I_Feel_Pretty | 0.263872 |
| Chappaquidick | 0.200920 |

</div>
