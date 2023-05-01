This benchmark is finding the root of Sin using the Newton's method and 4th order Taylor expansions of Sin and
Cosine. The input is any floating point number (Unit is Radians). The output is any floating point number again in Radians
iteratively getting closer to an x where sin(x) is 0.

FP64:\
Run 1:\
#define ANGLE_IN_RADIANS 3*M_PI/2
#define TOLERANCE 0.00001
#define N 4 \
No Mem Opt, 0 evaluations, At most 4 iterations\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1010940.json\
Number of Computation DAGs                      : 4\
Number of Computation Paths                     : 947248\
Number of Floating-Point Operations     : 84\
Average Amplification Factor            : 35.818750783251183\
Highest Amplification Factor: 1277.491489772223304\
Highest Condition Number: 332.671525853804724\
Highest Amplification Density: 126.919225749591376\
Longest Path Length                     : 41\
Time: 0.995104\
AF: 1277.491489772223304 (ULPErr: 11.000000; 2.000000 digits) of Node with AFId:2282077 WRT Input:%sin_x.0 through path: 
[(77, 43), (76, 43), (75, 43), (69, 31), (64, 24), (61, 43), (60, 43), (56, 43), (55, 43), (54, 43), (48, 31), (43, 24), 
(40, 43), (39, 43), (35, 43), (34, 43), (33, 43), (27, 31), (22, 24), (19, 43)]

Run 2:\
#define ANGLE_IN_RADIANS 3*M_PI/2
#define TOLERANCE 0.00001
#define N 2 \
No Mem Opt, 1000 evaluations\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1013237.json\
Number of Computation DAGs                      : 2\
Number of Computation Paths                     : 1124\
Number of Floating-Point Operations     : 42\
Average Amplification Factor            : 3.125371886080329\
Highest Amplification Factor: 9.473006082157880\
Highest Condition Number: 6.235772997398898\
Highest Amplification Density: 6.235772997398898\
Longest Path Length                     : 21\
Time: 0.333446\
AF: 9.473006082157880 (ULPErr: 4.000000; 1.000000 digits) of Node with AFId:2689 WRT Input:%sin_x.0 through path: [(35, 43), 
(34, 43), (33, 43), (27, 31), (22, 24), (19, 43)]

Run 3:\
#define ANGLE_IN_RADIANS 3*M_PI/2
#define TOLERANCE 0.00001
#define N 10 \
Mem Opt, 0 evaluations:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1016557.json\
Number of Computation DAGs                      : 7\
Number of Computation Paths                     : 14\
Number of Floating-Point Operations     : 147\
Average Amplification Factor            : 8019692659402362.000000000000000\
Highest Amplification Factor: 75781838470076704.000000000000000\
Highest Condition Number: 6932486167634994.000000000000000\
Highest Amplification Density: 1848337523660407.500000000000000\
Longest Path Length                     : 44\
Time: 0.000115\
AF: 75781838470076704.000000000000000 (ULPErr: 57.000000; 2.000000 digits) of Node with AFId:280 WRT Input:%sin_x.0 through path: 
[(140, 43), (139, 43), (138, 43), (132, 31), (127, 24), (124, 43), (123, 43), (119, 43), (118, 43), (117, 43), (111, 31), 
(106, 24), (103, 43), (102, 43), (98, 43), (97, 43), (96, 43), (90, 31), (85, 24), (82, 43), (81, 43), (77, 43), (76, 43), 
(75, 43), (69, 31), (64, 24), (61, 43), (60, 43), (56, 43), (55, 43), (54, 43), (48, 31), (43, 24), (40, 43), (39, 43), 
(35, 43), (34, 43), (33, 43), (27, 31), (22, 24), (19, 43)]

FP32:\
Run 1:\
#define ANGLE_IN_RADIANS 3*M_PI/2
#define TOLERANCE 0.00001
#define N 4 \
No Mem Opt, 0 evaluations, At most 4 iterations\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1011968.json\
Number of Computation DAGs                      : 4\
Number of Computation Paths                     : 947248\
Number of Floating-Point Operations     : 84\
Average Amplification Factor            : 35.818420894840990\
Highest Amplification Factor: 1277.479227974185051\
Highest Condition Number: 332.668276269488615\
Highest Amplification Density: 126.917988470508121\
Longest Path Length                     : 41\
Time: 0.786596\
AF: 1277.479227974185051 (ULPErr: 11.000000; 2.000000 digits) of Node with AFId:5147787 WRT Input:%conv22 through path: 
[(83, 48), (82, 43), (81, 43), (77, 43), (76, 43), (75, 43), (69, 31), (64, 24), (61, 43), (60, 43), (56, 43), (55, 43), 
(54, 43), (48, 31), (43, 24), (40, 43), (39, 43), (35, 43), (34, 43), (33, 43), (27, 31), (22, 24), (19, 43)]

Run 2:\
#define ANGLE_IN_RADIANS 3*M_PI/2
#define TOLERANCE 0.00001
#define N 2 \
No Mem Opt, 1000 evaluations\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1012816.json\
Number of Computation DAGs                      : 2\
Number of Computation Paths                     : 1124\
Number of Floating-Point Operations     : 42\
Average Amplification Factor            : 3.125372207324986\
Highest Amplification Factor: 9.473006432906255\
Highest Condition Number: 6.235773827153952\
Highest Amplification Density: 6.235773827153952\
Longest Path Length                     : 21\
Time: 0.283365\
AF: 9.473006432906255 (ULPErr: 4.000000; 1.000000 digits) of Node with AFId:2689 WRT Input:%conv22 through path: [(35, 43), 
(34, 43), (33, 43), (27, 31), (22, 24), (19, 43)]

Run 3:\
#define ANGLE_IN_RADIANS 3*M_PI/2
#define TOLERANCE 0.00001
#define N 10 \
Mem Opt, 0 evaluations:\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_1015453.json\
Number of Computation DAGs                      : 8\
Number of Computation Paths                     : 16\
Number of Floating-Point Operations     : 168\
Average Amplification Factor            : 0.000000000000000\
Highest Amplification Factor: inf\
Highest Condition Number: inf\
Highest Amplification Density: inf\
Longest Path Length                     : 45\
Time: 0.000140\
AF: inf (ULPErr: inf; inf digits) of Node with AFId:334 WRT Input:%conv22 through path: [(167, 48), (166, 43), (145, 43), 
(144, 43), (140, 43), (139, 43), (138, 43), (132, 31), (127, 24), (124, 43), (123, 43), (119, 43), (118, 43), (117, 43), 
(111, 31), (106, 24), (103, 43), (102, 43), (98, 43), (97, 43), (96, 43), (90, 31), (85, 24), (82, 43), (81, 43), (77, 43), 
(76, 43), (75, 43), (69, 31), (64, 24), (61, 43), (60, 43), (56, 43), (55, 43), (54, 43), (48, 31), (43, 24), (40, 43), 
(39, 43), (35, 43), (34, 43), (33, 43), (27, 31), (22, 24), (19, 43)]