# RUN: llvm-mc -triple riscv64 -mattr +c,-relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s -check-prefix CHECK-RVC
# RUN: llvm-mc -triple riscv64 -mattr +c,+relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s -check-prefix CHECK-RVC
# RUN: llvm-mc -triple riscv64 -mattr -c,-relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -triple riscv64 -mattr -c,+relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s

.balign 4
.byte 0

.balign 4
auipc a0, 0

# CHECK-RVC: 0000000000000000 <.text>:
# CHECK-RVC-NEXT: 0: 0000      	unimp
# CHECK-RVC-NEXT: 2: 0001      	nop
# CHECK-RVC-NEXT: 4: 00000517  	auipc	a0, 0x0

# CHECK: 0000000000000000 <.text>:
# CHECK-NEXT: 0: 0000      	<unknown>
# CHECK-NEXT: 2: 0001      	<unknown>
# CHECK-NEXT: 4: 00000517  	auipc	a0, 0x0
