# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 < %s | FileCheck %s --check-prefixes=CHECK,BTVER2

palignr     $1, %xmm0, %xmm2
palignr     $1, (%rax), %xmm2

phaddd      %xmm0, %xmm2
phaddd      (%rax), %xmm2

phaddsw     %xmm0, %xmm2
phaddsw     (%rax), %xmm2

phaddw      %xmm0, %xmm2
phaddw      (%rax), %xmm2

phsubd      %xmm0, %xmm2
phsubd      (%rax), %xmm2

phsubsw     %xmm0, %xmm2
phsubsw     (%rax), %xmm2

phsubw      %xmm0, %xmm2
phsubw      (%rax), %xmm2

pmaddubsw   %xmm0, %xmm2
pmaddubsw   (%rax), %xmm2

pmulhrsw    %xmm0, %xmm2
pmulhrsw    (%rax), %xmm2

pshufb      %xmm0, %xmm2
pshufb      (%rax), %xmm2

psignb      %xmm0, %xmm2
psignb      (%rax), %xmm2

psignd      %xmm0, %xmm2
psignd      (%rax), %xmm2

psignw      %xmm0, %xmm2
psignw      (%rax), %xmm2

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
# CHECK-NEXT:  -      -      -      -      -     0.99   0.01    -      -      -      -      -     1.00    -     	palignr	$1, %xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.01   0.99   1.00    -      -      -     1.00    -      -     	palignr	$1, (%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.99   0.01    -      -      -      -      -     1.00    -     	phaddd	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.01   0.99   1.00    -      -      -     1.00    -      -     	phaddd	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.99   0.01    -      -      -      -      -     1.00    -     	phaddsw	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.01   0.99   1.00    -      -      -     1.00    -      -     	phaddsw	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.99   0.01    -      -      -      -      -     1.00    -     	phaddw	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.01   0.99   1.00    -      -      -     1.00    -      -     	phaddw	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.99   0.01    -      -      -      -      -     1.00    -     	phsubd	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.01   0.99   1.00    -      -      -     1.00    -      -     	phsubd	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.99   0.01    -      -      -      -      -     1.00    -     	phsubsw	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.01   0.99   1.00    -      -      -     1.00    -      -     	phsubsw	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.99   0.01    -      -      -      -      -     1.00    -     	phsubw	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.01   0.99   1.00    -      -      -     1.00    -      -     	phsubw	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -      -     1.00   	pmaddubsw	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -     1.00    -      -      -      -      -     1.00   	pmaddubsw	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -      -     1.00   	pmulhrsw	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -     1.00    -      -      -      -      -     1.00   	pmulhrsw	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -      -      -     4.00    -     	pshufb	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00   1.00    -      -      -     4.00    -      -     	pshufb	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -     1.00    -     	psignb	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00   1.00    -      -      -     1.00    -      -     	psignb	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -     1.00    -     	psignd	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00   1.00    -      -      -     1.00    -      -     	psignd	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -     1.00    -     	psignw	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00   1.00    -      -      -     1.00    -      -     	psignw	(%rax), %xmm2
