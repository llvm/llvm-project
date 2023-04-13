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
No Mem Opt, 0 evaluations, at most Number of iterations (n)=4:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_822617.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 273166\
Number of Floating-Point Operations     : 96\
Average Amplification Factor            : 0.000020850962467\
Highest Amplification Factor: 13.000000\
Highest Condition Number: 13.000000\
Highest Amplification Density: 13.000000\
Longest Path Length                             : 32\
AF: 12.999999999999989 (ULPErr: 4.000000; 1.000000 digits) of Node with AFId:0 WRT Input:%y_0.addr.068 through path: [(0, 30)]

Run 2:\
No Mem Opt, 500 evaluations, n=2:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_824805.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 1201\
Number of Floating-Point Operations     : 48\
Average Amplification Factor            : 0.005557226775183\
Highest Amplification Factor: 3504.919894\
Highest Condition Number: 3504.919894\
Highest Amplification Density: 3504.919894\
Longest Path Length                             : 16\
AF: 3504.919893598597355 (ULPErr: 12.000000; 2.000000 digits) of Node with AFId:0 WRT Input:%y_0.addr.068 through path: [(0, 30)]

Run 3:\
Mem Opt enabled, 1 evaluation, n = 400:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_825817.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 9\
Number of Floating-Point Operations     : 9600\
Average Amplification Factor            : 0.055450422450926\
Highest Amplification Factor: 5571.978651\
Highest Condition Number: 3504.919894\
Highest Amplification Density: 3504.919894\
Longest Path Length                             : 800\
AF: 5571.978651494689075 (ULPErr: 13.000000; 2.000000 digits) of Node with AFId:5387 WRT Input:%x_0.addr.069 through path: 
[(2394, 42), (2392, 42), (2378, 30), (2366, 41), (2354, 30), (2342, 41), (2330, 30), (2318, 41), (2306, 30), (2294, 41), (2282, 30), 
(2270, 41), (2258, 30), (2246, 41), (2234, 30), (2222, 41), (2210, 30), (2198, 41), (2186, 30), (2174, 41), (2162, 30), (2150, 41), 
(2138, 30), (2126, 41), (2114, 30), (2102, 41), (2090, 30), (2078, 41), (2066, 30), (2054, 41), (2042, 30), (2030, 41), (2018, 30), 
(2006, 41), (1994, 30), (1982, 41), (1970, 30), (1958, 41), (1946, 30), (1934, 41), (1922, 30), (1910, 41), (1898, 30), (1886, 41), 
(1874, 30), (1862, 41), (1850, 30), (1838, 41), (1826, 30), (1814, 41), (1802, 30), (1790, 41), (1778, 30), (1766, 41), (1754, 30), 
(1742, 41), (1730, 30), (1718, 41), (1706, 30), (1694, 41), (1682, 30), (1670, 41), (1658, 30), (1646, 41), (1634, 30), (1622, 41), 
(1610, 30), (1598, 41), (1586, 30), (1574, 41), (1562, 30), (1550, 41), (1538, 30), (1526, 41), (1514, 30), (1502, 41), (1490, 30), 
(1478, 41), (1466, 30), (1454, 41), (1442, 30), (1430, 41), (1418, 30), (1406, 41), (1394, 30), (1382, 41), (1370, 30), (1358, 41), 
(1346, 30), (1334, 41), (1322, 30), (1310, 41), (1298, 30), (1286, 41), (1274, 30), (1262, 41), (1250, 30), (1238, 41), (1226, 30), 
(1214, 41), (1202, 30), (1190, 41), (1178, 30), (1166, 41), (1154, 30), (1142, 41), (1130, 30), (1118, 41), (1106, 30), (1094, 41), 
(1082, 30), (1070, 41), (1058, 30), (1046, 41), (1034, 30), (1022, 41), (1010, 30), (998, 41), (986, 30), (974, 41), (962, 30), (950, 41), 
(938, 30), (926, 41), (914, 30), (902, 41), (890, 30), (878, 41), (866, 30), (854, 41), (842, 30), (830, 41), (818, 30), (806, 41), 
(794, 30), (782, 41), (770, 30), (758, 41), (746, 30), (734, 41), (722, 30), (710, 41), (698, 30), (686, 41), (674, 30), (662, 41), 
(650, 30), (638, 41), (626, 30), (614, 41), (602, 30), (590, 41), (578, 30), (566, 41), (554, 30), (542, 41), (530, 30), (518, 41), 
(506, 30), (494, 41), (482, 30), (470, 41), (458, 30), (446, 41), (434, 30), (422, 41), (410, 30), (398, 41), (386, 30), (374, 41), 
(362, 30), (350, 41), (338, 30), (326, 41), (314, 30), (302, 41), (290, 30), (278, 41), (266, 30), (254, 41), (242, 30), (230, 41), 
(218, 30), (206, 41), (194, 30), (182, 41), (170, 30), (158, 41), (146, 30), (134, 41), (122, 30), (110, 41), (98, 30), (86, 41), 
(74, 30), (62, 41), (50, 30), (38, 41), (26, 30), (14, 41), (2, 30)]

FP32:\
Run 3:\
Mem Opt enabled, 1 evaluation, n = 400:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_826846.json\
Number of Computation DAGs                      : 3\
Number of Computation Paths                     : 9\
Number of Floating-Point Operations     : 9600\
Average Amplification Factor            : 0.055450497935575\
Highest Amplification Factor: 5577.129919\
Highest Condition Number: 3504.788839\
Highest Amplification Density: 3504.788839\
Longest Path Length                             : 800\
AF: 5577.129918978525893 (ULPErr: 13.000000; 2.000000 digits) of Node with AFId:5386 WRT Input:%conv through path: [(2394, 42), 
(2392, 42), (2378, 30), (2366, 41), (2354, 30), (2342, 41), (2330, 30), (2318, 41), (2306, 30), (2294, 41), (2282, 30), 
(2270, 41), (2258, 30), (2246, 41), (2234, 30), (2222, 41), (2210, 30), (2198, 41), (2186, 30), (2174, 41), (2162, 30), 
(2150, 41), (2138, 30), (2126, 41), (2114, 30), (2102, 41), (2090, 30), (2078, 41), (2066, 30), (2054, 41), (2042, 30), 
(2030, 41), (2018, 30), (2006, 41), (1994, 30), (1982, 41), (1970, 30), (1958, 41), (1946, 30), (1934, 41), (1922, 30), 
(1910, 41), (1898, 30), (1886, 41), (1874, 30), (1862, 41), (1850, 30), (1838, 41), (1826, 30), (1814, 41), (1802, 30), 
(1790, 41), (1778, 30), (1766, 41), (1754, 30), (1742, 41), (1730, 30), (1718, 41), (1706, 30), (1694, 41), (1682, 30), 
(1670, 41), (1658, 30), (1646, 41), (1634, 30), (1622, 41), (1610, 30), (1598, 41), (1586, 30), (1574, 41), (1562, 30), 
(1550, 41), (1538, 30), (1526, 41), (1514, 30), (1502, 41), (1490, 30), (1478, 41), (1466, 30), (1454, 41), (1442, 30), 
(1430, 41), (1418, 30), (1406, 41), (1394, 30), (1382, 41), (1370, 30), (1358, 41), (1346, 30), (1334, 41), (1322, 30),
(1310, 41), (1298, 30), (1286, 41), (1274, 30), (1262, 41), (1250, 30), (1238, 41), (1226, 30), (1214, 41), (1202, 30),
(1190, 41), (1178, 30), (1166, 41), (1154, 30), (1142, 41), (1130, 30), (1118, 41), (1106, 30), (1094, 41), (1082, 30), 
(1070, 41), (1058, 30), (1046, 41), (1034, 30), (1022, 41), (1010, 30), (998, 41), (986, 30), (974, 41), (962, 30), (950, 41), 
(938, 30), (926, 41), (914, 30), (902, 41), (890, 30), (878, 41), (866, 30), (854, 41), (842, 30), (830, 41), (818, 30), 
(806, 41), (794, 30), (782, 41), (770, 30), (758, 41), (746, 30), (734, 41), (722, 30), (710, 41), (698, 30), (686, 41), 
(674, 30), (662, 41), (650, 30), (638, 41), (626, 30), (614, 41), (602, 30), (590, 41), (578, 30), (566, 41), (554, 30), 
(542, 41), (530, 30), (518, 41), (506, 30), (494, 41), (482, 30), (470, 41), (458, 30), (446, 41), (434, 30), (422, 41), 
(410, 30), (398, 41), (386, 30), (374, 41), (362, 30), (350, 41), (338, 30), (326, 41), (314, 30), (302, 41), (290, 30), 
(278, 41), (266, 30), (254, 41), (242, 30), (230, 41), (218, 30), (206, 41), (194, 30), (182, 41), (170, 30), (158, 41), 
(146, 30), (134, 41), (122, 30), (110, 41), (98, 30), (86, 41), (74, 30), (62, 41), (50, 30), (38, 41), (26, 30), (14, 41), 
(2, 30)]

