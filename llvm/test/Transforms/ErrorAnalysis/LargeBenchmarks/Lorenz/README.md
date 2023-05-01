The Lorenz system is a system of ordinary differential equations notable for having chaotic solutions for certain 
parameter values and initial conditions.

The model is a system of three ordinary differential equations now known as the Lorenz equations:



The equations relate the properties of a two-dimensional fluid layer uniformly warmed from below and cooled from above. 
In particular, the equations describe the rate of change of three quantities with respect to time: x is proportional to 
the rate of convection, y to the horizontal temperature variation, and z to the vertical temperature variation. The 
constants σ, ρ, and β are system parameters proportional to the Prandtl number, Rayleigh number, and certain physical 
dimensions of the layer itself.

One normally assumes that the parameters σ, ρ, and β are positive. Lorenz used the values σ = 10, β = 8 / 3 and ρ = 28. 
The system exhibits chaotic behavior for these (and nearby) values.

If ρ < 1 then there is only one equilibrium point, which is at the origin. This point corresponds to no convection. All 
orbits converge to the origin, which is a global attractor, when ρ < 1.

# Results
FP64:\
Run 1:\
TYPE RHO = 12;
TYPE SIGMA = 10;
TYPE BETA = 8/3;
#define x 0.797150265052314
#define y 0.797377767475103
#define z 0.020565719820157
#define n 5 \
No Mem Opt, 0 evaluations, at most Number of iterations (n)=5:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1019509.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 4111369\
Number of Floating-Point Operations     : 120\
Average Amplification Factor            : 0.000002039090818\
Highest Amplification Factor: 3504.919893598597355\
Highest Condition Number: 3504.919893598597355\
Highest Amplification Density: 3504.919893598597355\
Longest Path Length                     : 40\
Time: 1.914457\
AF: 3504.919893598597355 (ULPErr: 12.000000; 2.000000 digits) of Node with AFId:0 WRT Input:%y_0.addr.066 through path: [(0, 35)]


Run 2:\
TYPE RHO = 12;
TYPE SIGMA = 10;
TYPE BETA = 8/3;
#define x 0.797150265052314
#define y 0.797377767475103
#define z 0.020565719820157
#define n 5 \
No Mem Opt, 500 evaluations\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1020979.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 18142\
Number of Floating-Point Operations     : 72\
Average Amplification Factor            : 0.000332750996378\
Highest Amplification Factor: 1505221.985676885582507\
Highest Condition Number: 1505221.985676885582507\
Highest Amplification Density: 1505221.985676885582507\
Longest Path Length                     : 24\
Time: 2.769342\
AF: 1505221.985676885582507 (ULPErr: 21.000000; 2.000000 digits) of Node with AFId:1 WRT Input:%x_0.addr.067 through path: [(15840, 35)]

Run 3:\
TYPE RHO = 12;
TYPE SIGMA = 10;
TYPE BETA = 8/3;
#define x 0.797150265052314
#define y 0.797377767475103
#define z 0.020565719820157
#define n 5 \
Mem Opt enabled, 0 evaluations, n = 400:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1022344.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 9\
Number of Floating-Point Operations     : 9600\
Average Amplification Factor            : 0.055450422450926\
Highest Amplification Factor: 5571.978651494689075\
Highest Condition Number: 3504.919893598597355\
Highest Amplification Density: 3504.919893598597355\
Longest Path Length                     : 800\
Time: 0.010277\
AF: 5571.978651494689075 (ULPErr: 13.000000; 2.000000 digits) of Node with AFId:5387 WRT Input:%x_0.addr.067 through path: 
[(2394, 47), (2392, 47), (2378, 35), (2366, 46), (2354, 35), (2342, 46), (2330, 35), (2318, 46), (2306, 35), (2294, 46), 
(2282, 35), (2270, 46), (2258, 35), (2246, 46), (2234, 35), (2222, 46), (2210, 35), (2198, 46), (2186, 35), (2174, 46), 
(2162, 35), (2150, 46), (2138, 35), (2126, 46), (2114, 35), (2102, 46), (2090, 35), (2078, 46), (2066, 35), (2054, 46), 
(2042, 35), (2030, 46), (2018, 35), (2006, 46), (1994, 35), (1982, 46), (1970, 35), (1958, 46), (1946, 35), (1934, 46), 
(1922, 35), (1910, 46), (1898, 35), (1886, 46), (1874, 35), (1862, 46), (1850, 35), (1838, 46), (1826, 35), (1814, 46), 
(1802, 35), (1790, 46), (1778, 35), (1766, 46), (1754, 35), (1742, 46), (1730, 35), (1718, 46), (1706, 35), (1694, 46), 
(1682, 35), (1670, 46), (1658, 35), (1646, 46), (1634, 35), (1622, 46), (1610, 35), (1598, 46), (1586, 35), (1574, 46), 
(1562, 35), (1550, 46), (1538, 35), (1526, 46), (1514, 35), (1502, 46), (1490, 35), (1478, 46), (1466, 35), (1454, 46), 
(1442, 35), (1430, 46), (1418, 35), (1406, 46), (1394, 35), (1382, 46), (1370, 35), (1358, 46), (1346, 35), (1334, 46), 
(1322, 35), (1310, 46), (1298, 35), (1286, 46), (1274, 35), (1262, 46), (1250, 35), (1238, 46), (1226, 35), (1214, 46), 
(1202, 35), (1190, 46), (1178, 35), (1166, 46), (1154, 35), (1142, 46), (1130, 35), (1118, 46), (1106, 35), (1094, 46), 
(1082, 35), (1070, 46), (1058, 35), (1046, 46), (1034, 35), (1022, 46), (1010, 35), (998, 46), (986, 35), (974, 46), (96
2, 35), (950, 46), (938, 35), (926, 46), (914, 35), (902, 46), (890, 35), (878, 46), (866, 35), (854, 46), (842, 35), 
(830, 46), (818, 35), (806, 46), (794, 35), (782, 46), (770, 35), (758, 46), (746, 35), (734, 46), (722, 35), (710, 46), 
(698, 35), (686, 46), (674, 35), (662, 46), (650, 35), (638, 46), (626, 35), (614, 46), (602, 35), (590, 46), (578, 35), 
(566, 46), (554, 35), (542, 46), (530, 35), (518, 46), (506, 35), (494, 46), (482, 35), (470, 46), (458, 35), (446, 46), 
(434, 35), (422, 46), (410, 35), (398, 46), (386, 35), (374, 46), (362, 35), (350, 46), (338, 35), (326, 46), (314, 35), 
(302, 46), (290, 35), (278, 46), (266, 35), (254, 46), (242, 35), (230, 46), (218, 35), (206, 46), (194, 35), (182, 46), 
(170, 35), (158, 46), (146, 35), (134, 46), (122, 35), (110, 46), (98, 35), (86, 46), (74, 35), (62, 46), (50, 35), (38, 46), 
(26, 35), (14, 46), (2, 35)]

FP32:\
Run 1:\
TYPE RHO = 12;
TYPE SIGMA = 10;
TYPE BETA = 8/3;
#define x 0.797150265052314
#define y 0.797377767475103
#define z 0.020565719820157
#define n 2 \
No Mem Opt, 0 evaluations, at most Number of iterations (n)=4:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1018526.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 4111369\
Number of Floating-Point Operations     : 120\
Average Amplification Factor            : 0.000002039090824\
Highest Amplification Factor: 3504.788839402672238\
Highest Condition Number: 3504.788839402672238\
Highest Amplification Density: 3504.788839402672238\
Longest Path Length                     : 40\
Time: 1.661388\
AF: 3504.788839402672238 (ULPErr: 12.000000; 2.000000 digits) of Node with AFId:0 WRT Input:%y_0.addr.095 through path: [(0, 35)]

Run 2:\
TYPE RHO = 12;
TYPE SIGMA = 10;
TYPE BETA = 8/3;
#define x 0.797150265052314
#define y 0.797377767475103
#define z 0.020565719820157
#define n 5 \
No Mem Opt, 500 evaluations\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1020612.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 18142\
Number of Floating-Point Operations     : 72\
Average Amplification Factor            : 0.000332764603218\
Highest Amplification Factor: 1672300.000000000000000\
Highest Condition Number: 1672300.000000000000000\
Highest Amplification Density: 1672300.000000000000000\
Longest Path Length                     : 24\
Time: 2.711837\
AF: 1672300.000000000000000 (ULPErr: 21.000000; 2.000000 digits) of Node with AFId:1 WRT Input:%x_0.addr.096 through path: [(13248, 35)]

Run 3:\
TYPE RHO = 12;
TYPE SIGMA = 10;
TYPE BETA = 8/3;
#define x 0.797150265052314
#define y 0.797377767475103
#define z 0.020565719820157
#define n 5 \
Mem Opt enabled, 0 evaluation, n = 400:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1021931.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 9\
Number of Floating-Point Operations     : 9600\
Average Amplification Factor            : 0.055450523701227\
Highest Amplification Factor: 5576.455432794858098\
Highest Condition Number: 3504.788839402672238\
Highest Amplification Density: 3504.788839402672238\
Longest Path Length                     : 800\
Time: 0.008846\
AF: 5576.455432794858098 (ULPErr: 13.000000; 2.000000 digits) of Node with AFId:5387 WRT Input:%conv through path: [(2394, 47), 
(2392, 47), (2378, 35), (2366, 46), (2354, 35), (2342, 46), (2330, 35), (2318, 46), (2306, 35), (2294, 46), (2282, 35), 
(2270, 46), (2258, 35), (2246, 46), (2234, 35), (2222, 46), (2210, 35), (2198, 46), (2186, 35), (2174, 46), (2162, 35), 
(2150, 46), (2138, 35), (2126, 46), (2114, 35), (2102, 46), (2090, 35), (2078, 46), (2066, 35), (2054, 46), (2042, 35), 
(2030, 46), (2018, 35), (2006, 46), (1994, 35), (1982, 46), (1970, 35), (1958, 46), (1946, 35), (1934, 46), (1922, 35), 
(1910, 46), (1898, 35), (1886, 46), (1874, 35), (1862, 46), (1850, 35), (1838, 46), (1826, 35), (1814, 46), (1802, 35), 
(1790, 46), (1778, 35), (1766, 46), (1754, 35), (1742, 46), (1730, 35), (1718, 46), (1706, 35), (1694, 46), (1682, 35), 
(1670, 46), (1658, 35), (1646, 46), (1634, 35), (1622, 46), (1610, 35), (1598, 46), (1586, 35), (1574, 46), (1562, 35), 
(1550, 46), (1538, 35), (1526, 46), (1514, 35), (1502, 46), (1490, 35), (1478, 46), (1466, 35), (1454, 46), (1442, 35), 
(1430, 46), (1418, 35), (1406, 46), (1394, 35), (1382, 46), (1370, 35), (1358, 46), (1346, 35), (1334, 46), (1322, 35), 
(1310, 46), (1298, 35), (1286, 46), (1274, 35), (1262, 46), (1250, 35), (1238, 46), (1226, 35), (1214, 46), (1202, 35), 
(1190, 46), (1178, 35), (1166, 46), (1154, 35), (1142, 46), (1130, 35), (1118, 46), (1106, 35), (1094, 46), (1082, 35), 
(1070, 46), (1058, 35), (1046, 46), (1034, 35), (1022, 46), (1010, 35), (998, 46), (986, 35), (974, 46), (962, 35), (950, 46), 
(938, 35), (926, 46), (914, 35), (902, 46), (890, 35), (878, 46), (866, 35), (854, 46), (842, 35), (830, 46), (818, 35), 
(806, 46), (794, 35), (782, 46), (770, 35), (758, 46), (746, 35), (734, 46), (722, 35), (710, 46), (698, 35), (686, 46), 
(674, 35), (662, 46), (650, 35), (638, 46), (626, 35), (614, 46), (602, 35), (590, 46), (578, 35), (566, 46), (554, 35), 
(542, 46), (530, 35), (518, 46), (506, 35), (494, 46), (482, 35), (470, 46), (458, 35), (446, 46), (434, 35), (422, 46), 
(410, 35), (398, 46), (386, 35), (374, 46), (362, 35), (350, 46), (338, 35), (326, 46), (314, 35), (302, 46), (290, 35), 
(278, 46), (266, 35), (254, 46), (242, 35), (230, 46), (218, 35), (206, 46), (194, 35), (182, 46), (170, 35), (158, 46), 
(146, 35), (134, 46), (122, 35), (110, 46), (98, 35), (86, 46), (74, 35), (62, 46), (50, 35), (38, 46), (26, 35), (14, 46), 
(2, 35)]