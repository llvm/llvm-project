# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=300 -timeline -timeline-max-iterations=3 < %s | FileCheck %s

vmulps   %xmm0, %xmm1, %xmm2
vhaddps  %xmm2, %xmm2, %xmm3
vhaddps  %xmm3, %xmm3, %xmm4

# CHECK:      Iterations:     300
# CHECK-NEXT: Instructions:   900
# CHECK-NEXT: Total Cycles:   610
# CHECK-NEXT: Dispatch Width: 2
# CHECK-NEXT: IPC:            1.48


# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
# CHECK-NEXT:  1      2     1.00                    	vmulps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT:  1      3     1.00                    	vhaddps	%xmm2, %xmm2, %xmm3
# CHECK-NEXT:  1      3     1.00                    	vhaddps	%xmm3, %xmm3, %xmm4


# CHECK:      Resources:
# CHECK-NEXT: [0] - JALU0
# CHECK-NEXT: [1] - JALU1
# CHECK-NEXT: [2] - JDiv
# CHECK-NEXT: [3] - JFPA
# CHECK-NEXT: [4] - JFPM
# CHECK-NEXT: [5] - JFPU0
# CHECK-NEXT: [6] - JFPU1
# CHECK-NEXT: [7] - JLAGU
# CHECK-NEXT: [8] - JMul
# CHECK-NEXT: [9] - JSAGU
# CHECK-NEXT: [10] - JSTC
# CHECK-NEXT: [11] - JVALU0
# CHECK-NEXT: [12] - JVALU1
# CHECK-NEXT: [13] - JVIMUL

# CHECK:      Resource pressure per iteration:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   
# CHECK-NEXT:  -      -      -     2.00   1.00   2.00   1.00    -      -      -      -      -      -      -     

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   	Instructions:
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00    -      -      -      -      -      -      -     	vmulps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	vhaddps	%xmm2, %xmm2, %xmm3
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	vhaddps	%xmm3, %xmm3, %xmm4


# CHECK:      Timeline view:
# CHECK-NEXT:      	          012345
# CHECK-NEXT: Index	0123456789

# CHECK:      [0,0]	DeeER.    .    .	vmulps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [0,1]	D==eeeER  .    .	vhaddps	%xmm2, %xmm2, %xmm3
# CHECK-NEXT: [0,2]	.D====eeeER    .	vhaddps	%xmm3, %xmm3, %xmm4

# CHECK:      [1,0]	.DeeE-----R    .	vmulps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [1,1]	. D=eeeE--R    .	vhaddps	%xmm2, %xmm2, %xmm3
# CHECK-NEXT: [1,2]	. D====eeeER   .	vhaddps	%xmm3, %xmm3, %xmm4

# CHECK:      [2,0]	.  DeeE----R   .	vmulps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [2,1]	.  D====eeeER  .	vhaddps	%xmm2, %xmm2, %xmm3
# CHECK-NEXT: [2,2]	.   D======eeeER	vhaddps	%xmm3, %xmm3, %xmm4

# CHECK:      Average Wait times (based on the timeline view):
# CHECK-NEXT: [0]: Executions
# CHECK-NEXT: [1]: Average time spent waiting in a scheduler's queue
# CHECK-NEXT: [2]: Average time spent waiting in a scheduler's queue while ready
# CHECK-NEXT: [3]: Average time elapsed from WB until retire stage

# CHECK:            [0]    [1]    [2]    [3]
# CHECK-NEXT: 0.     3     1.0    1.0    3.0  	vmulps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: 1.     3     3.3    0.7    0.7  	vhaddps	%xmm2, %xmm2, %xmm3
# CHECK-NEXT: 2.     3     5.7    0.0    0.0  	vhaddps	%xmm3, %xmm3, %xmm4
