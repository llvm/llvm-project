# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 < %s | FileCheck %s

vcvtph2ps   %xmm0, %xmm2
vcvtph2ps   (%rax), %xmm2

vcvtph2ps   %xmm0, %ymm2
vcvtph2ps   (%rax), %ymm2

vcvtps2ph   $0, %xmm0, %xmm2
vcvtps2ph   $0, %xmm0, (%rax)

vcvtps2ph   $0, %ymm0, %xmm2
vcvtps2ph   $0, %ymm0, (%rax)

# CHECK:      Iterations:     100
# CHECK-NEXT: Instructions:   800
# CHECK-NEXT: Total Cycles:   1503
# CHECK-NEXT: Dispatch Width: 2
# CHECK-NEXT: IPC:            0.53

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
# CHECK-NEXT:  1      3     1.00                    	vcvtph2ps	%xmm0, %xmm2
# CHECK-NEXT:  1      8     1.00    *               	vcvtph2ps	(%rax), %xmm2
# CHECK-NEXT:  2      3     2.00                    	vcvtph2ps	%xmm0, %ymm2
# CHECK-NEXT:  2      8     2.00    *               	vcvtph2ps	(%rax), %ymm2
# CHECK-NEXT:  1      3     1.00                    	vcvtps2ph	$0, %xmm0, %xmm2
# CHECK-NEXT:  1      3     1.00           *        	vcvtps2ph	$0, %xmm0, (%rax)
# CHECK-NEXT:  3      6     2.00                    	vcvtps2ph	$0, %ymm0, %xmm2
# CHECK-NEXT:  3      11    2.00           *        	vcvtps2ph	$0, %ymm0, (%rax)

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
# CHECK-NEXT:  -      -      -     2.00   2.00    -     12.00  2.00    -     2.00   12.00   -      -      -

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   	Instructions:
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -     1.00    -      -      -     	vcvtph2ps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00   1.00    -      -     1.00    -      -      -     	vcvtph2ps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     2.00    -      -      -     2.00    -      -      -     	vcvtph2ps	%xmm0, %ymm2
# CHECK-NEXT:  -      -      -      -      -      -     2.00   1.00    -      -     2.00    -      -      -     	vcvtph2ps	(%rax), %ymm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -     1.00    -      -      -     	vcvtps2ph	$0, %xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -     1.00   1.00    -      -      -     	vcvtps2ph	$0, %xmm0, (%rax)
# CHECK-NEXT:  -      -      -     1.86   0.14    -     2.00    -      -      -     2.00    -      -      -     	vcvtps2ph	$0, %ymm0, %xmm2
# CHECK-NEXT:  -      -      -     0.14   1.86    -     2.00    -      -     1.00   2.00    -      -      -     	vcvtps2ph	$0, %ymm0, (%rax)

