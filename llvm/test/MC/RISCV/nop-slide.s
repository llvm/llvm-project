# RUN: llvm-mc -triple riscv64 -mattr +c,-relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -triple riscv64 -mattr +c,+relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -triple riscv64 -mattr -c,-relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -triple riscv64 -mattr -c,+relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s

.balign 4
.byte 0

.balign 4
auipc a0, 0

# CHECK: 0000000000000000 <.text>:
# CHECK-NEXT: 0: 00 00 01 00   .word   0x00010000
# CHECK-NEXT: 4: 00000517  	auipc	a0, 0x0
