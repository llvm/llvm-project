# RUN: llvm-mc %s -triple=riscv32 -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 \
# RUN:     | FileCheck -check-prefixes=CHECK-S,CHECK-S-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv64 \
# RUN:     | FileCheck -check-prefixes=CHECK-S,CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S                 Match the .s output with aliases enabled
# CHECK-S-NOALIAS         Match the .s output with aliases disabled
# CHECK-OBJ               Match the objdumped object output with aliases enabled
# CHECK-OBJ-NOALIAS       Match the objdumped object output with aliases enabled
# CHECK-S-OBJ             Match both the .s and objdumped object output with
#                         aliases enabled
# CHECK-S-OBJ-NOALIAS     Match both the .s and objdumped object output with
#                         aliases disabled

# TODO la
# TODO lb lh lw
# TODO sb sh sw

# CHECK-S-OBJ-NOALIAS: addi zero, zero, 0
# CHECK-S-OBJ: nop
nop

# CHECK-S-OBJ-NOALIAS: addi t6, zero, 0
# CHECK-S-OBJ: li t6, 0
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

# CHECK-S-NOALIAS: beq a0, zero, 512
# CHECK-S: beqz a0, 512
# CHECK-OBJ-NOALIAS: beq a0, zero, 0x22c
# CHECK-OBJ: beqz a0, 0x22c
beqz x10, 512
# CHECK-S-NOALIAS: bne a1, zero, 1024
# CHECK-S: bnez a1, 1024
# CHECK-OBJ-NOALIAS: bne a1, zero, 0x430
# CHECK-OBJ: bnez a1, 0x430
bnez x11, 1024
# CHECK-S-NOALIAS: bge zero, a2, 4
# CHECK-S: blez a2, 4
# CHECK-OBJ-NOALIAS: bge zero, a2, 0x38
# CHECK-OBJ: blez a2, 0x38
blez x12, 4
# CHECK-S-NOALIAS: bge a3, zero, 8
# CHECK-S: bgez a3, 8
# CHECK-OBJ-NOALIAS: bge a3, zero, 0x40
# CHECK-OBJ: bgez a3, 0x40
bgez x13, 8
# CHECK-S-NOALIAS: blt a4, zero, 12
# CHECK-S: bltz a4, 12
# CHECK-OBJ-NOALIAS: blt a4, zero, 0x48
# CHECK-OBJ: bltz a4, 0x48
bltz x14, 12
# CHECK-S-NOALIAS: blt zero, a5, 16
# CHECK-S: bgtz a5, 16
# CHECK-OBJ-NOALIAS: blt zero, a5, 0x50
# CHECK-OBJ: bgtz a5, 0x50
bgtz x15, 16

# Always output the canonical mnemonic for the pseudo branch instructions.
# CHECK-S-NOALIAS: blt a6, a5, 20
# CHECK-S: blt a6, a5, 20
# CHECK-OBJ-NOALIAS: blt a6, a5, 0x58
# CHECK-OBJ: blt a6, a5, 0x58
bgt x15, x16, 20
# CHECK-S-NOALIAS: bge a7, a6, 24
# CHECK-S: bge a7, a6, 24
# CHECK-OBJ-NOALIAS: bge a7, a6, 0x60
# CHECK-OBJ: bge a7, a6, 0x60
ble x16, x17, 24
# CHECK-S-NOALIAS: bltu s2, a7, 28
# CHECK-S: bltu s2, a7, 28
# CHECK-OBJ-NOALIAS: bltu s2, a7, 0x68
# CHECK-OBJ: bltu s2, a7, 0x68
bgtu x17, x18, 28
# CHECK-S-NOALIAS: bgeu s3, s2, 32
# CHECK-S: bgeu s3, s2, 32
# CHECK-OBJ-NOALIAS: bgeu s3, s2, 0x70
# CHECK-OBJ: bgeu s3, s2, 0x70
bleu x18, x19, 32

# CHECK-S-NOALIAS: jal zero, 2044
# CHECK-S: j 2044
# CHECK-OBJ-NOALIAS: jal zero, 0x850
# CHECK-OBJ: j 0x850
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
# CHECK-S-NOALIAS: jal ra, 2040
# CHECK-S: jal 2040
# CHECK-OBJ-NOALIAS: jal ra, 0x85c
# CHECK-OBJ: jal 0x85c
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
# CHECK-S-OBJ-NOALIAS: jalr zero, 0(s11)
# CHECK-S-OBJ: jr s11
jr (x27)
# CHECK-S-OBJ-NOALIAS: jalr ra, 0(t3)
# CHECK-S-OBJ: jalr t3
jalr (x28)
# CHECK-S-OBJ-NOALIAS: jalr t4, 0(t5)
# CHECK-S-OBJ: jalr t4, t5
jalr x29, (x30)

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

# CHECK-S-OBJ-NOALIAS: csrrs s0, siselect, zero
# CHECK-S-OBJ: csrr s0, siselect
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

# CHECK-S-OBJ-NOALIAS: csrrwi zero, siselect, 15
# CHECK-S-OBJ: csrwi siselect, 15
csrwi 0x150, 0xf
# CHECK-S-OBJ-NOALIAS: csrrsi zero, 4095, 16
# CHECK-S-OBJ: csrsi 4095, 16
csrsi 0xfff, 0x10
# CHECK-S-OBJ-NOALIAS: csrrci zero, sscratch, 17
# CHECK-S-OBJ: csrci sscratch, 17
csrci 0x140, 0x11

# CHECK-S-OBJ-NOALIAS: csrrwi zero, siselect, 7
# CHECK-S-OBJ: csrwi siselect, 7
csrw 0x150, 7
# CHECK-S-OBJ-NOALIAS: csrrsi zero, siselect, 7
# CHECK-S-OBJ: csrsi siselect, 7
csrs 0x150, 7
# CHECK-S-OBJ-NOALIAS: csrrci zero, siselect, 7
# CHECK-S-OBJ: csrci siselect, 7
csrc 0x150, 7

# CHECK-S-OBJ-NOALIAS: csrrwi t0, siselect, 15
# CHECK-S-OBJ: csrrwi t0, siselect, 15
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
