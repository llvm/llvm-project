# Results
FP64:\
Run 1:\
#define N 2
#define A00 20
#define A01 1
#define A10 1
#define A11 3
#define B0 1
#define B1 2
const TYPE TOLERANCE = 0.0000000001;
const TYPE NEARZERO = 0.0000000001;\
No Mem Opt, 0 evaluations, at most 1 iteration\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_338939.json\
Number of Computation DAGs                      : 9\
Number of Computation Paths                     : 233\
Number of Floating-Point Operations     : 45\
Average Amplification Factor            : 0.000000000000000\
Highest Amplification Factor: inf\
Highest Condition Number: inf\
Highest Amplification Density: inf\
Longest Path Length                     : 11\
Time: 0.000171\
AF: inf (ULPErr: inf; inf digits) of Node with AFId:517 WRT Input:%R_0.0 through path: [(41, 197), (39, 196), (36, 185), (32, 176), (31, 176)]

Run 2:\
#define N 2
#define A00 20
#define A01 1
#define A10 1
#define A11 3
#define B0 1
#define B1 2
const TYPE TOLERANCE = 0.0000000001;
const TYPE NEARZERO = 0.0000000001;
No Mem Opt, 1000 evaluations\


Run 3:\
Same as Run 1

FP32:\
Run 1:\
#define N 2
#define A00 20
#define A01 1
#define A10 1
#define A11 3
#define B0 1
#define B1 2
const TYPE TOLERANCE = 0.0000000001;
const TYPE NEARZERO = 0.0000000001;
No Mem Opt, 0 evaluations, at most 1 iteration\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_340309.json\
Number of Computation DAGs                      : 9\
Number of Computation Paths                     : 233\
Number of Floating-Point Operations     : 45\
Average Amplification Factor            : 644432.112722700228915\
Highest Amplification Factor: 14617757.635924801230431\
Highest Condition Number: 12320682.435993760824203\
Highest Amplification Density: 12320682.435993760824203\
Longest Path Length                     : 11\
Time: 0.000222\
AF: 14617757.635924801230431 (ULPErr: 24.000000; 2.000000 digits) of Node with AFId:511 WRT Input:%A_0_0 through path: [(40, 196), (26, 169), (25, 169)]

Run 2:\
#define N 2
#define A00 20
#define A01 1
#define A10 1
#define A11 3
#define B0 1
#define B1 2
const TYPE TOLERANCE = 0.0000000001;
const TYPE NEARZERO = 0.0000000001;
No Mem Opt, 1000 evaluations\
Ranked AF Paths written to file: .fAF_logs/SortedAFs_revenant_343825.json\
Number of Computation DAGs                      : 6\
Number of Computation Paths                     : 278\
Number of Floating-Point Operations     : 44\
Average Amplification Factor            : 13318.256524838177938\
Highest Amplification Factor: 1453027308.683725357055664\
Highest Condition Number: 1298913299.015750408172607\
Highest Amplification Density: 1298913299.015750408172607\
Longest Path Length                     : 11\
Time: 0.051152\
AF: 1453027308.683725357055664 (ULPErr: 31.000000; 2.000000 digits) of Node with AFId:491 WRT Input:%P_1.0 through path: [(27295, 197), (27285, 170), (27284, 170), (27283, 170)]
Crashed...

Run 3:\
Same as Run 1