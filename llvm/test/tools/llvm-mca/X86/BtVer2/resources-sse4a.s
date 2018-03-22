# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 < %s | FileCheck %s --check-prefixes=CHECK,BTVER2

extrq       %xmm0, %xmm2
extrq       $22, $2, %xmm2

insertq     %xmm0, %xmm2
insertq     $22, $22, %xmm0, %xmm2

movntsd     %xmm0, (%rax)
movntss     %xmm0, (%rax)

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

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   	Instructions:
# CHECK-NEXT:  -      -      -      -      -     0.99   0.01    -      -      -      -     0.49   0.51    -     	extrq	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -     0.50   0.50    -     	extrq	$22, $2, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -      -     2.00   2.00    -     	insertq	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -     2.00   2.00    -     	insertq	$22, $22, %xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -     1.00   1.00    -      -      -     	movntsd	%xmm0, (%rax)
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -     1.00   1.00    -      -      -     	movntss	%xmm0, (%rax)
