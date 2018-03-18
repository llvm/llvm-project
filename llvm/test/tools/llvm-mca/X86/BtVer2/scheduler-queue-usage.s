# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=1 -verbose < %s | FileCheck %s

vmulps (%rsi), %xmm0, %xmm0
add  %rsi, %rsi

# CHECK:      Scheduler's queue usage:
# CHECK-NEXT: JALU01,  1/20
# CHECK-NEXT: JFPU01,  1/18
# CHECK-NEXT: JLSAGU,  1/12

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
# CHECK-NEXT:  -     1.00    -      -     1.00    -     1.00   1.00    -      -      -      -      -      -     

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   	Instructions:
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00   1.00    -      -      -      -      -      -     	vmulps	(%rsi), %xmm0, %xmm0
# CHECK-NEXT:  -     1.00    -      -      -      -      -      -      -      -      -      -      -      -     	addq	%rsi, %rsi
