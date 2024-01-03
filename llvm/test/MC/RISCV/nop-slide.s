# RUN: llvm-mc -triple riscv64 -mattr +c,-relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s -check-prefix CHECK-RVC-NORELAX
# RUN: llvm-mc -triple riscv64 -mattr +c,+relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s -check-prefix CHECK-RVC-RELAX
# RUN: llvm-mc -triple riscv64 -mattr -c,-relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -triple riscv64 -mattr -c,+relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s

.balign 4
.byte 0

.balign 4
auipc a0, 0

# CHECK-RVC-NORELAX: 0000000000000000 <.text>:
# CHECK-RVC-NORELAX-NEXT: 0: 00 00        	unimp
# CHECK-RVC-NORELAX-NEXT: 2: 01 00        	nop
# CHECK-RVC-NORELAX-NEXT: 4: 17 05 00 00  	auipc	a0, 0x0

# CHECK-RVC-RELAX: 0000000000000000 <.text>:
# CHECK-RVC-RELAX-NEXT:   0: 01 00        	nop
# CHECK-RVC-RELAX-NEXT:   2: 00 01        	addi	s0, sp, 0x80
# CHECK-RVC-RELAX-NEXT:   4: 00 17        	addi	s0, sp, 0x3a0
# CHECK-RVC-RELAX-NEXT:   6: 05 00        	c.nop	0x1
# CHECK-RVC-RELAX-NEXT:   8: 00           	<unknown>

# CHECK: 0000000000000000 <.text>:
# CHECK-NEXT: 0: 00 00        	<unknown>
# CHECK-NEXT: 2: 00 00        	<unknown>
# CHECK-NEXT: 4: 17 05 00 00  	auipc	a0, 0x0
