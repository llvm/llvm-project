# RUN: llvm-mc -triple x86_64 %s -o - | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o %t
# RUN: llvm-readelf -SsX -x data %t | FileCheck %s
# RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=DIS

# ASM: .section	.text,"ax",@progbits,unique,4294967293
# ASM: f:

# ASM: .section	.text,"ax",@progbits,unique,4294967294
# ASM: g:

# CHECK:       Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK:        text             PROGBITS        0000000000000000 {{.*}} 000000 00  AX  0   0  4
# CHECK-NEXT:  .text             PROGBITS        0000000000000000 {{.*}} 000004 00  AX  0   0  1
# CHECK-NEXT:  .text             PROGBITS        0000000000000000 {{.*}} 000001 00  AX  0   0  1

# CHECK:       0000000000000001     0 NOTYPE  LOCAL  DEFAULT     3 (.text)  f2
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT     3 (.text)  f
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT     4 (.text)  g

# CHECK:      Hex dump of section 'data':
# CHECK-NEXT: 0x00000000 03000000 06000000                   .
# CHECK-EMPTY:
# CHECK-NEXT: Hex dump of section 'data':
# CHECK-NEXT: 0x00000000 04000000                            .

# DIS:      Disassembly of section .text:
# DIS-EMPTY:
# DIS-NEXT: 0000000000000000 <f>:
# DIS-NEXT:        0: 90                            nop
# DIS-EMPTY:
# DIS-NEXT: 0000000000000001 <f2>:
# DIS-NEXT:        1: 90                            nop
# DIS-NEXT:        2: cc                            int3
# DIS-NEXT:        3: cc                            int3
# DIS-EMPTY:
# DIS-NEXT: Disassembly of section .text:
# DIS-EMPTY:
# DIS-NEXT: 0000000000000000 <g>:
# DIS-NEXT:        0: 90                            nop

	.section	.text,"ax",@progbits,unique, 4294967293
        .globl	f
f:
        nop

	.section	.text,"ax",@progbits,unique, 4294967294
        .globl	g
g:
        nop

.section .text,"ax",@progbits, unique, 4294967293
f2:
  nop

.pushsection data,"a",@progbits,unique,3
.long 3
.popsection
int3
.pushsection data,"a",@progbits,unique,4
.long 4
.popsection
.pushsection data,"a",@progbits,unique,3
.long 6
.popsection
int3
