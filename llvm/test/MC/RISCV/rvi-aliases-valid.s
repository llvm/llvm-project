# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 \
# RUN:     | FileCheck -check-prefixes=CHECK-S,CHECK-S-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases\
# RUN:     | FileCheck -check-prefixes=CHECK-S-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv64 \
# RUN:     | FileCheck -check-prefixes=CHECK-S,CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -r -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d -r -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-S-OBJ %s

# COM: The following check prefixes are used in this test:
# COM: CHECK-S                 Match the .s output with aliases enabled
# COM: CHECK-S-NOALIAS         Match the .s output with aliases disabled
# COM: CHECK-OBJ               Match the objdumped object output with aliases enabled
# COM: CHECK-OBJ-NOALIAS       Match the objdumped object output with aliases enabled
# COM: CHECK-S-OBJ             Match both the .s and objdumped object output with
# COM:                         aliases enabled
# COM: CHECK-S-OBJ-NOALIAS     Match both the .s and objdumped object output with
# COM:                         aliases disabled

# TODO la
# TODO lb lh lw
# TODO sb sh sw

# CHECK-S-OBJ-NOALIAS: addi zero, zero, 0
# CHECK-S-OBJ: nop
nop

# CHECK-S-OBJ-NOALIAS: addi t6, zero, 0
# CHECK-S-OBJ: mv t6, zero
mv x31, zero
# CHECK-S-OBJ-NOALIAS: addi a2, a3, 0
# CHECK-S-OBJ: mv a2, a3
move a2,a3
# CHECK-S-OBJ-NOALIAS: xori t6, ra, -1
# CHECK-S-OBJ: not t6, ra
not x31, x1
# CHECK-S-OBJ-NOALIAS: sub t6, zero, ra
# CHECK-S-OBJ: neg t6, ra
neg x31, x1
# CHECK-S-OBJ-NOALIAS: sltiu t6, ra, 1
# CHECK-S-OBJ: seqz t6, ra
seqz x31, x1
# CHECK-S-OBJ-NOALIAS: sltu t6, zero, ra
# CHECK-S-OBJ: snez t6, ra
snez x31, x1
# CHECK-S-OBJ-NOALIAS: slt t6, ra, zero
# CHECK-S-OBJ: sltz t6, ra
sltz x31, x1
# CHECK-S-OBJ-NOALIAS: slt t6, zero, ra
# CHECK-S-OBJ: sgtz t6, ra
sgtz x31, x1

# CHECK-S-OBJ-NOALIAS: slt ra, gp, sp
# CHECK-S-OBJ: slt ra, gp, sp
sgt x1, x2, x3
# CHECK-S-OBJ-NOALIAS: sltu tp, t1, t0
# CHECK-S-OBJ: sltu tp, t1, t0
sgtu x4, x5, x6

# CHECK-S-OBJ-NOALIAS: beq a0, zero, 512
# CHECK-S-OBJ: beqz a0, 512
beqz x10, 512
# CHECK-S-OBJ-NOALIAS: bne a1, zero, 1024
# CHECK-S-OBJ: bnez a1, 1024
bnez x11, 1024
# CHECK-S-OBJ-NOALIAS: bge zero, a2, 4
# CHECK-S-OBJ: blez a2, 4
blez x12, 4
# CHECK-S-OBJ-NOALIAS: bge a3, zero, 8
# CHECK-S-OBJ: bgez a3, 8
bgez x13, 8
# CHECK-S-OBJ-NOALIAS: blt a4, zero, 12
# CHECK-S-OBJ: bltz a4, 12
bltz x14, 12
# CHECK-S-OBJ-NOALIAS: blt zero, a5, 16
# CHECK-S-OBJ: bgtz a5, 16
bgtz x15, 16

# Always output the canonical mnemonic for the pseudo branch instructions.
# CHECK-S-OBJ-NOALIAS: blt a6, a5, 20
# CHECK-S-OBJ: blt a6, a5, 20
bgt x15, x16, 20
# CHECK-S-OBJ-NOALIAS: bge a7, a6, 24
# CHECK-S-OBJ: bge a7, a6, 24
ble x16, x17, 24
# CHECK-S-OBJ-NOALIAS: bltu s2, a7, 28
# CHECK-S-OBJ: bltu s2, a7, 28
bgtu x17, x18, 28
# CHECK-S-OBJ-NOALIAS: bgeu s3, s2, 32
# CHECK-S-OBJ: bgeu s3, s2, 32
bleu x18, x19, 32

# CHECK-S-OBJ-NOALIAS: jal zero, 2044
# CHECK-S-OBJ: j 2044
j 2044
# CHECK-S-NOALIAS: jal zero, foo
# CHECK-S: j foo
# CHECK-OBJ-NOALIAS: jal zero, 0
# CHECK-OBJ: j 0
# CHECK-OBJ: R_RISCV_JAL foo
j foo
# CHECK-S-NOALIAS: jal zero, a0
# CHECK-S: j a0
# CHECK-OBJ-NOALIAS: jal zero, 0
# CHECK-OBJ: j 0
# CHECK-OBJ: R_RISCV_JAL a0
j a0
# CHECK-S-NOALIAS: [[LABEL:.L[[:alnum:]_]+]]:
# CHECK-S-NOALIAS-NEXT: jal zero, [[LABEL]]
# CHECK-S: [[LABEL:.L[[:alnum:]_]+]]:
# CHECK-S-NEXT: j [[LABEL]]
# CHECK-OBJ-NOALIAS: jal zero, 0
# CHECK-OBJ: j 0
j .
# CHECK-S-OBJ-NOALIAS: jal ra, 2040
# CHECK-S-OBJ: jal 2040
jal 2040
# CHECK-S-NOALIAS: jal ra, foo
# CHECK-S: jal foo
# CHECK-OBJ-NOALIAS: jal ra, 0
# CHECK-OBJ: jal 0
# CHECK-OBJ: R_RISCV_JAL foo
jal foo
# CHECK-S-NOALIAS: jal ra, a0
# CHECK-S: jal a0
# CHECK-OBJ-NOALIAS: jal ra, 0
# CHECK-OBJ: jal 0
# CHECK-OBJ: R_RISCV_JAL a0
jal a0
# CHECK-S-OBJ-NOALIAS: jalr zero, 0(s4)
# CHECK-S-OBJ: jr s4
jr x20
# CHECK-S-OBJ-NOALIAS: jalr zero, 6(s5)
# CHECK-S-OBJ: jr 6(s5)
jr 6(x21)
# CHECK-S-OBJ-NOALIAS: jalr zero, 7(s6)
# CHECK-S-OBJ: jr 7(s6)
jr x22, 7
# CHECK-S-OBJ-NOALIAS: jalr ra, 0(s4)
# CHECK-S-OBJ: jalr s4
jalr x20
# CHECK-S-OBJ-NOALIAS: jalr ra, 8(s5)
# CHECK-S-OBJ: jalr 8(s5)
jalr 8(x21)
# CHECK-S-OBJ-NOALIAS: jalr s6, 0(s7)
# CHECK-S-OBJ: jalr s6, s7
jalr x22, x23
# CHECK-S-OBJ-NOALIAS: jalr ra, 9(s8)
# CHECK-S-OBJ: jalr 9(s8)
jalr x24, 9
# CHECK-S-OBJ-NOALIAS: jalr s9, 11(s10)
# CHECK-S-OBJ: jalr s9, 11(s10)
jalr x25, x26, 11
# CHECK-S-OBJ-NOALIAS: jalr zero, 0(ra)
# CHECK-S-OBJ: ret
ret
# TODO call
# TODO tail

# CHECK-S-OBJ-NOALIAS: fence iorw, iorw
# CHECK-S-OBJ: fence
fence

# CHECK-S-OBJ-NOALIAS: csrrs s10, instret, zero
# CHECK-S-OBJ: rdinstret s10
rdinstret x26
# CHECK-S-OBJ-NOALIAS: csrrs s8, cycle, zero
# CHECK-S-OBJ: rdcycle s8
rdcycle x24
# CHECK-S-OBJ-NOALIAS: csrrs s9, time, zero
# CHECK-S-OBJ: rdtime s9
rdtime x25

# CHECK-S-OBJ-NOALIAS: csrrs  s0, 336, zero
# CHECK-S-OBJ: csrr s0, 336
csrr x8, 0x150
# CHECK-S-OBJ-NOALIAS: csrrw zero, sscratch, s1
# CHECK-S-OBJ: csrw sscratch, s1
csrw 0x140, x9
# CHECK-S-OBJ-NOALIAS: csrrs zero, 4095, s6
# CHECK-S-OBJ: csrs 4095, s6
csrs 0xfff, x22
# CHECK-S-OBJ-NOALIAS: csrrc zero, 4095, s7
# CHECK-S-OBJ: csrc 4095, s7
csrc 0xfff, x23

# CHECK-S-OBJ-NOALIAS: csrrwi zero, 336, 15
# CHECK-S-OBJ: csrwi 336, 15
csrwi 0x150, 0xf
# CHECK-S-OBJ-NOALIAS: csrrsi zero, 4095, 16
# CHECK-S-OBJ: csrsi 4095, 16
csrsi 0xfff, 0x10
# CHECK-S-OBJ-NOALIAS: csrrci zero, sscratch, 17
# CHECK-S-OBJ: csrci sscratch, 17
csrci 0x140, 0x11

# CHECK-S-OBJ-NOALIAS: csrrwi zero, 336, 7
# CHECK-S-OBJ: csrwi 336, 7
csrw 0x150, 7
# CHECK-S-OBJ-NOALIAS: csrrsi zero, 336, 7
# CHECK-S-OBJ: csrsi 336, 7
csrs 0x150, 7
# CHECK-S-OBJ-NOALIAS: csrrci zero, 336, 7
# CHECK-S-OBJ: csrci 336, 7
csrc 0x150, 7

# CHECK-S-OBJ-NOALIAS: csrrwi t0, 336, 15
# CHECK-S-OBJ: csrrwi t0, 336, 15
csrrw t0, 0x150, 0xf
# CHECK-S-OBJ-NOALIAS: csrrsi t0, 4095, 16
# CHECK-S-OBJ: csrrsi t0, 4095, 16
csrrs t0, 0xfff, 0x10
# CHECK-S-OBJ-NOALIAS: csrrci t0, sscratch, 17
# CHECK-S-OBJ: csrrci t0, sscratch, 17
csrrc t0, 0x140, 0x11

# CHECK-S-OBJ-NOALIAS: sfence.vma zero, zero
# CHECK-S-OBJ: sfence.vma
sfence.vma
# CHECK-S-OBJ-NOALIAS: sfence.vma a0, zero
# CHECK-S-OBJ: sfence.vma a0
sfence.vma a0

# The following aliases are accepted as input but the canonical form
# of the instruction will always be printed.
# CHECK-S-OBJ-NOALIAS: addi a2, a3, 4
# CHECK-S-OBJ: addi a2, a3, 4
add a2, a3, 4
# CHECK-S-OBJ-NOALIAS: andi a2, a3, 4
# CHECK-S-OBJ: andi a2, a3, 4
and a2, a3, 4
# CHECK-S-OBJ-NOALIAS: xori a2, a3, 4
# CHECK-S-OBJ: xori a2, a3, 4
xor a2, a3, 4
# CHECK-S-OBJ-NOALIAS: ori a2, a3, 4
# CHECK-S-OBJ: ori a2, a3, 4
or a2, a3, 4
# CHECK-S-OBJ-NOALIAS: slli a2, a3, 4
# CHECK-S-OBJ: slli a2, a3, 4
sll a2, a3, 4
# CHECK-S-OBJ-NOALIAS: srli a2, a3, 4
# CHECK-S-OBJ: srli a2, a3, 4
srl a2, a3, 4
# CHECK-S-OBJ-NOALIAS: srai a2, a3, 4
# CHECK-S-OBJ: srai a2, a3, 4
sra a2, a3, 4
# CHECK-S-OBJ-NOALIAS: slti a2, a3, 4
# CHECK-S-OBJ: slti a2, a3, 4
slt a2, a3, 4
# CHECK-S-OBJ-NOALIAS: sltiu a2, a3, 4
# CHECK-S-OBJ: sltiu a2, a3, 4
sltu a2, a3, 4

# CHECK-S-OBJ-NOALIAS: ebreak
# CHECK-S-OBJ: ebreak
sbreak

# CHECK-S-OBJ-NOALIAS: ecall
# CHECK-S-OBJ: ecall
scall
