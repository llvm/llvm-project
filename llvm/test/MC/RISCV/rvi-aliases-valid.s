# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases\
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -riscv-no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d -riscv-no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

# TODO la
# TODO lb lh lw
# TODO sb sh sw

# CHECK-INST: addi zero, zero, 0
# CHECK-ALIAS: nop
nop
# TODO li
# CHECK-INST: addi t6, zero, 0
# CHECK-ALIAS: mv t6, zero
mv x31, zero
# CHECK-INST: xori t6, ra, -1
# CHECK-ALIAS: not t6, ra
not x31, x1
# CHECK-INST: sub t6, zero, ra
# CHECK-ALIAS: neg t6, ra
neg x31, x1
# CHECK-INST: sltiu t6, ra, 1
# CHECK-ALIAS: seqz t6, ra
seqz x31, x1
# CHECK-INST: sltu t6, zero, ra
# CHECK-ALIAS: snez t6, ra
snez x31, x1
# CHECK-INST: slt t6, ra, zero
# CHECK-ALIAS: sltz t6, ra
sltz x31, x1
# CHECK-INST: slt t6, zero, ra
# CHECK-ALIAS: sgtz t6, ra
sgtz x31, x1

# CHECK-INST: beq a0, zero, 512
# CHECK-ALIAS: beqz a0, 512
beqz x10, 512
# CHECK-INST: bne a1, zero, 1024
# CHECK-ALIAS: bnez a1, 1024
bnez x11, 1024
# CHECK-INST: bge zero, a2, 4
# CHECK-ALIAS: blez a2, 4
blez x12, 4
# CHECK-INST: bge a3, zero, 8
# CHECK-ALIAS: bgez a3, 8
bgez x13, 8
# CHECK-INST: blt a4, zero, 12
# CHECK-ALIAS: bltz a4, 12
bltz x14, 12
# CHECK-INST: blt zero, a5, 16
# CHECK-ALIAS: bgtz a5, 16
bgtz x15, 16

# Always output the canonical mnemonic for the pseudo branch instructions.
# CHECK-INST: blt a6, a5, 20
# CHECK-ALIAS: blt a6, a5, 20
bgt x15, x16, 20
# CHECK-INST: bge a7, a6, 24
# CHECK-ALIAS: bge a7, a6, 24
ble x16, x17, 24
# CHECK-INST: bltu s2, a7, 28
# CHECK-ALIAS: bltu s2, a7, 28
bgtu x17, x18, 28
# CHECK-INST: bgeu s3, s2, 32
# CHECK-ALIAS: bgeu s3, s2, 32
bleu x18, x19, 32

# CHECK-INST: jal zero, 2044
# CHECK-ALIAS: j 2044
j 2044
# CHECK-INST: jal ra, 2040
# CHECK-ALIAS: jal 2040
jal 2040
# CHECK-INST: jalr zero, s4, 0
# CHECK-ALIAS: jr s4
jr x20
# CHECK-INST: jalr ra, s5, 0
# CHECK-ALIAS: jalr s5
jalr x21
# CHECK-INST: jalr zero, ra, 0
# CHECK-ALIAS: ret
ret
# TODO call
# TODO tail

# CHECK-INST: fence iorw, iorw
# CHECK-ALIAS: fence
fence

# CHECK-INST: csrrs s10, 3074, zero
# CHECK-ALIAS: rdinstret s10
rdinstret x26
# CHECK-INST: csrrs s8, 3072, zero
# CHECK-ALIAS: rdcycle s8
rdcycle x24
# CHECK-INST: csrrs s9, 3073, zero
# CHECK-ALIAS: rdtime s9
rdtime x25

# CHECK-INST: csrrs  s0, 336, zero
# CHECK-ALIAS: csrr s0, 336
csrr x8, 0x150
# CHECK-INST: csrrw zero, 320, s1
# CHECK-ALIAS: csrw 320, s1
csrw 0x140, x9
# CHECK-INST: csrrs zero, 4095, s6
# CHECK-ALIAS: csrs 4095, s6
csrs 0xfff, x22
# CHECK-INST: csrrc zero, 4095, s7
# CHECK-ALIAS: csrc 4095, s7
csrc 0xfff, x23

# CHECK-INST: csrrwi zero, 336, 15
# CHECK-ALIAS: csrwi 336, 15
csrwi 0x150, 0xf
# CHECK-INST: csrrsi zero, 4095, 16
# CHECK-ALIAS: csrsi 4095, 16
csrsi 0xfff, 0x10
# CHECK-INST: csrrci zero, 320, 17
# CHECK-ALIAS: csrci 320, 17
csrci 0x140, 0x11

# CHECK-INST: sfence.vma zero, zero
# CHECK-ALIAS: sfence.vma
sfence.vma
# CHECK-INST: sfence.vma a0, zero
# CHECK-ALIAS: sfence.vma a0
sfence.vma a0
