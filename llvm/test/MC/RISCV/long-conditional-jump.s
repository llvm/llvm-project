# RUN: llvm-mc -filetype=obj -triple=riscv64 %s \
# RUN:     | llvm-objdump -d -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c %s \
# RUN:     | llvm-objdump -d -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST-C %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s \
# RUN:     | llvm-objdump -dr -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST-RELAX %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s \
# RUN:     | llvm-objdump -dr -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST-C-RELAX %s

       .text
       .type   test,@function
test:
# CHECK-INST:         beq     a0, a1, 0x8
# CHECK-INST-NEXT:    jal     zero, 0x1458
# CHECK-INST-C:       beq     a0, a1, 0x8
# CHECK-INST-C-NEXT:  jal     zero, 0x1458
# CHECK-INST-RELAX:         beq     a0, a1, 0x8
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L1
# CHECK-INST-C-RELAX:       beq     a0, a1, 0x8
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L1
   bne a0, a1, .L1
   call relax
.fill 1300-2, 4, 0
.L1:
   ret
# CHECK-INST:         bne     a0, a1, 0x1464
# CHECK-INST-NEXT:    jal     zero, 0x28b4
# CHECK-INST-C:       bne     a0, a1, 0x1462
# CHECK-INST-C-NEXT:  jal     zero, 0x28b2
# CHECK-INST-RELAX:         bne     a0, a1, 0x1464
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L2
# CHECK-INST-C-RELAX:       bne     a0, a1, 0x1462
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L2
   beq a0, a1, .L2
.fill 1300, 4, 0
.L2:
   ret
# CHECK-INST:         bge     a0, a1, 0x28c0
# CHECK-INST-NEXT:    jal     zero, 0x3d10
# CHECK-INST-C:       bge     a0, a1, 0x28bc
# CHECK-INST-C-NEXT:  jal     zero, 0x3d0c
# CHECK-INST-RELAX:         bge     a0, a1, 0x28c0
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L3
# CHECK-INST-C-RELAX:       bge     a0, a1, 0x28bc
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L3
   blt a0, a1, .L3
.fill 1300, 4, 0
.L3:
   ret
# CHECK-INST:         blt     a0, a1, 0x3d1c
# CHECK-INST-NEXT:    jal     zero, 0x516c
# CHECK-INST-C:       blt     a0, a1, 0x3d16
# CHECK-INST-C-NEXT:  jal     zero, 0x5166
# CHECK-INST-RELAX:         blt     a0, a1, 0x3d1c
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L4
# CHECK-INST-C-RELAX:       blt     a0, a1, 0x3d16
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L4
   bge a0, a1, .L4
.fill 1300, 4, 0
.L4:
   ret
# CHECK-INST:         bgeu    a0, a1, 0x5178
# CHECK-INST-NEXT:    jal     zero, 0x65c8
# CHECK-INST-C:       bgeu    a0, a1, 0x5170
# CHECK-INST-C-NEXT:  jal     zero, 0x65c0
# CHECK-INST-RELAX:         bgeu    a0, a1, 0x5178
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L5
# CHECK-INST-C-RELAX:       bgeu    a0, a1, 0x5170
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L5
   bltu a0, a1, .L5
.fill 1300, 4, 0
.L5:
   ret
# CHECK-INST:         bltu    a0, a1, 0x65d4
# CHECK-INST-NEXT:    jal     zero, 0x7a24
# CHECK-INST-C:       bltu    a0, a1, 0x65ca
# CHECK-INST-C-NEXT:  jal     zero, 0x7a1a
# CHECK-INST-RELAX:         bltu    a0, a1, 0x65d4
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L6
# CHECK-INST-C-RELAX:       bltu    a0, a1, 0x65ca
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L6
   bgeu a0, a1, .L6
.fill 1300, 4, 0
.L6:
   ret
# CHECK-INST:         bne     a0, zero, 0x7a30
# CHECK-INST-NEXT:    jal     zero, 0x8e80
# CHECK-INST-C:       c.bnez  a0, 0x7a22
# CHECK-INST-C-NEXT:  jal     zero, 0x8e72
# CHECK-INST-RELAX:         bne     a0, zero, 0x7a30
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L7
# CHECK-INST-C-RELAX:       c.bnez  a0, 0x7a22
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L7
   beqz a0, .L7
.fill 1300, 4, 0
.L7:
   ret
# CHECK-INST:         bne     zero, a0, 0x8e8c
# CHECK-INST-NEXT:    jal     zero, 0xa2dc
# CHECK-INST-C:       c.bnez  a0, 0x8e7a
# CHECK-INST-C-NEXT:  jal     zero, 0xa2ca
# CHECK-INST-RELAX:         bne     zero, a0, 0x8e8c
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L8
# CHECK-INST-C-RELAX:       c.bnez  a0, 0x8e7a
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L8
   beq x0, a0, .L8
.fill 1300, 4, 0
.L8:
   ret
# CHECK-INST:         beq     a0, zero, 0xa2e8
# CHECK-INST-NEXT:    jal     zero, 0xb738
# CHECK-INST-C:       c.beqz  a0, 0xa2d2
# CHECK-INST-C-NEXT:  jal     zero, 0xb722
# CHECK-INST-RELAX:         beq     a0, zero, 0xa2e8
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L9
# CHECK-INST-C-RELAX:       c.beqz  a0, 0xa2d2
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L9
   bnez a0, .L9
.fill 1300, 4, 0
.L9:
   ret
# CHECK-INST:         beq     a6, zero, 0xb744
# CHECK-INST-NEXT:    jal     zero, 0xcb94
# CHECK-INST-C:       beq     a6, zero, 0xb72c
# CHECK-INST-C-NEXT:  jal     zero, 0xcb7c
# CHECK-INST-RELAX:         beq     a6, zero, 0xb744
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
# CHECK-INST-RELAX-NEXT:    R_RISCV_JAL .L10
# CHECK-INST-C-RELAX:       beq     a6, zero, 0xb72c
# CHECK-INST-C-RELAX-NEXT:  jal     zero, {{.*}}
# CHECK-INST-C-RELAX-NEXT:  R_RISCV_JAL .L10
   bnez x16, .L10
.fill 1300, 4, 0
.L10:
   ret
.Lfunc_end0:
       .size   test, .Lfunc_end0-test
