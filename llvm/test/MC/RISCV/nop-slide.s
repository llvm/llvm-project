# RUN: llvm-mc -triple riscv64 -mattr +c,-relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s -check-prefix CHECK-RVC-NORELAX
# RUN: llvm-mc -triple riscv64 -mattr +c,+relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s -check-prefix CHECK-RVC-RELAX
# RUN: llvm-mc -triple riscv64 -mattr -c,-relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -triple riscv64 -mattr -c,+relax -filetype obj -o - %s | llvm-objdump -d - | FileCheck %s

.balign 4
.byte 0

.balign 4
auipc a0, 0

# CHECK-RVC-NORELAX: 0000000000000000 <.text>:
# CHECK-RVC-NORELAX-NEXT: 0: 00 00 01 00 .word 0x00010000
# CHECK-RVC-NORELAX-NEXT: 4: 00000517  	auipc	a0, 0x0

# CHECK-RVC-RELAX: 0000000000000000 <.text>:
# CHECK-RVC-RELAX-NEXT:   0: 0001      	nop
# CHECK-RVC-RELAX-NEXT:   2: 00 01      .short 0x0100
# CHECK-RVC-RELAX-NEXT:   4: 00         .byte 0x00
# CHECK-RVC-RELAX-NEXT:   5: 00000517  	auipc	a0, 0x0

# CHECK: 0000000000000000 <.text>:
# CHECK-NEXT: 0: 00 00 00 00 .word 0x00000000
# CHECK-NEXT: 4: 00000517  	auipc	a0, 0x0
