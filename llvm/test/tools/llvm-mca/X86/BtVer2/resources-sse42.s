# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -instruction-tables < %s | FileCheck %s --check-prefixes=CHECK,BTVER2

crc32b      %al, %ecx
crc32b      (%rax), %ecx

crc32l      %eax, %ecx
crc32l      (%rax), %ecx

crc32w      %ax, %ecx
crc32w      (%rax), %ecx

crc32b      %al, %rcx
crc32b      (%rax), %rcx

crc32q      %rax, %rcx
crc32q      (%rax), %rcx

pcmpestri   $1, %xmm0, %xmm2
pcmpestri   $1, (%rax), %xmm2

pcmpestrm   $1, %xmm0, %xmm2
pcmpestrm   $1, (%rax), %xmm2

pcmpistri   $1, %xmm0, %xmm2
pcmpistri   $1, (%rax), %xmm2

pcmpistrm   $1, %xmm0, %xmm2
pcmpistrm   $1, (%rax), %xmm2

pcmpgtq     %xmm0, %xmm2
pcmpgtq     (%rax), %xmm2

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
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -      -      -      -      -      -      -      -     	crc32b	%al, %ecx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -     1.00    -      -      -      -      -      -     	crc32b	(%rax), %ecx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -      -      -      -      -      -      -      -     	crc32l	%eax, %ecx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -     1.00    -      -      -      -      -      -     	crc32l	(%rax), %ecx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -      -      -      -      -      -      -      -     	crc32w	%ax, %ecx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -     1.00    -      -      -      -      -      -     	crc32w	(%rax), %ecx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -      -      -      -      -      -      -      -     	crc32b	%al, %rcx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -     1.00    -      -      -      -      -      -     	crc32b	(%rax), %rcx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -      -      -      -      -      -      -      -     	crc32q	%rax, %rcx
# CHECK-NEXT: 2.00   2.00    -      -      -      -      -     1.00    -      -      -      -      -      -     	crc32q	(%rax), %rcx
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   2.00    -     2.00    -     3.00   7.00    -     	pcmpestri	$1, %xmm0, %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   3.00    -     2.00    -     3.00   7.00    -     	pcmpestri	$1, (%rax), %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   2.00    -     2.00    -     3.00   7.00    -     	pcmpestrm	$1, %xmm0, %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   3.00    -     2.00    -     3.00   7.00    -     	pcmpestrm	$1, (%rax), %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00    -      -      -      -      -     2.00    -     	pcmpistri	$1, %xmm0, %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   1.00    -      -      -      -     2.00    -     	pcmpistri	$1, (%rax), %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00    -      -      -      -      -     2.00    -     	pcmpistrm	$1, %xmm0, %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   1.00    -      -      -      -     2.00    -     	pcmpistrm	$1, (%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.50   0.50    -      -      -      -     0.50   0.50    -     	pcmpgtq	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.50   0.50   1.00    -      -      -     0.50   0.50    -     	pcmpgtq	(%rax), %xmm2
