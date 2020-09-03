# RUN: llvm-mc -triple riscv32 -mattr=+c -show-encoding < %s \
# RUN: | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -show-encoding \
# RUN: -riscv-no-aliases <%s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj < %s \
# RUN: | llvm-objdump  --triple=riscv32 --mattr=+c -d - \
# RUN: | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj < %s \
# RUN: | llvm-objdump  --triple=riscv32 --mattr=+c -d -M no-aliases - \
# RUN: | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# RUN: llvm-mc -triple riscv64 -mattr=+c -show-encoding < %s \
# RUN: | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -show-encoding \
# RUN: -riscv-no-aliases <%s | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj < %s \
# RUN: | llvm-objdump  --triple=riscv64 --mattr=+c -d - \
# RUN: | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj < %s \
# RUN: | llvm-objdump  --triple=riscv64 --mattr=+c -d -M no-aliases - \
# RUN: | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# CHECK-BYTES: 2e 85
# CHECK-ALIAS: add a0, zero, a1
# CHECK-INST: c.mv a0, a1
# CHECK: # encoding:  [0x2e,0x85]
addi a0, a1, 0

# CHECK-BYTES: e0 1f
# CHECK-ALIAS: addi s0, sp, 1020
# CHECK-INST: c.addi4spn s0, sp, 1020
# CHECK: # encoding:  [0xe0,0x1f]
addi s0, sp, 1020

# CHECK-BYTES: e0 5f
# CHECK-ALIAS: lw s0, 124(a5)
# CHECK-INST: c.lw s0, 124(a5)
# CHECK: # encoding: [0xe0,0x5f]
lw s0, 124(a5)

# CHECK-BYTES: e0 df
# CHECK-ALIAS: sw s0, 124(a5)
# CHECK-INST: c.sw s0, 124(a5)
# CHECK: # encoding: [0xe0,0xdf]
sw s0, 124(a5)

# CHECK-BYTES: 01 00
# CHECK-ALIAS: nop
# CHECK-INST: c.nop
# CHECK: # encoding: [0x01,0x00]
nop

# CHECK-BYTES: 81 10
# CHECK-ALIAS: addi ra, ra, -32
# CHECK-INST: c.addi ra, -32
# CHECK: # encoding:  [0x81,0x10]
addi ra, ra, -32

# CHECK-BYTES: 85 50
# CHECK-ALIAS: addi ra, zero, -31
# CHECK-INST: c.li ra, -31
# CHECK: # encoding: [0x85,0x50]
addi ra, zero, -31

# CHECK-BYTES: 39 71
# CHECK-ALIAS: addi sp, sp, -64
# CHECK-INST: c.addi16sp sp, -64
# CHECK:  # encoding: [0x39,0x71]
addi sp, sp, -64

# CHECK-BYTES: fd 61
# CHECK-ALIAS: lui gp, 31
# CHECK-INST: c.lui gp, 31
# CHECK: # encoding:  [0xfd,0x61]
lui gp, 31

# CHECK-BYTES: 7d 80
# CHECK-ALIAS: srli s0, s0, 31
# CHECK-INST: c.srli s0, 31
# CHECK: # encoding:  [0x7d,0x80]
srli s0, s0, 31

# CHECK-BYTES: 7d 84
# CHECK-ALIAS: srai s0, s0, 31
# CHECK-INST: c.srai s0, 31
# CHECK: # encoding: [0x7d,0x84]
srai s0, s0, 31

# CHECK-BYTES: 7d 88
# CHECK-ALIAS: andi s0, s0, 31
# CHECK-INST: c.andi s0, 31
# CHECK: # encoding: [0x7d,0x88]
andi s0, s0, 31

# CHECK-BYTES: 1d 8c
# CHECK-ALIAS: sub s0, s0, a5
# CHECK-INST: c.sub s0, a5
# CHECK: # encoding: [0x1d,0x8c]
sub s0, s0, a5

# CHECK-BYTES: 3d 8c
# CHECK-ALIAS: xor s0, s0, a5
# CHECK-INST: c.xor s0, a5
# CHECK: # encoding: [0x3d,0x8c]
xor s0, s0, a5

# CHECK-BYTES: 3d 8c
# CHECK-ALIAS: xor s0, s0, a5
# CHECK-INST: c.xor s0, a5
# CHECK: # encoding: [0x3d,0x8c]
xor s0, a5, s0

# CHECK-BYTES: 5d 8c
# CHECK-ALIAS: or s0, s0, a5
# CHECK-INST: c.or s0, a5
# CHECK: # encoding:  [0x5d,0x8c]
or s0, s0, a5

# CHECK-BYTES: 45 8c
# CHECK-ALIAS: or s0, s0, s1
# CHECK-INST: c.or s0, s1
# CHECK:  # encoding: [0x45,0x8c]
or  s0, s1, s0

# CHECK-BYTES: 7d 8c
# CHECK-ALIAS: and s0, s0, a5
# CHECK-INST: c.and s0, a5
# CHECK: # encoding: [0x7d,0x8c]
and s0, s0, a5

# CHECK-BYTES: 7d 8c
# CHECK-ALIAS: and s0, s0, a5
# CHECK-INST: c.and s0, a5
# CHECK: # encoding: [0x7d,0x8c]
and s0, a5, s0

# CHECK-BYTES: 01 b0
# CHECK-ALIAS: j -2048
# CHECK-INST: c.j -2048
# CHECK:  # encoding: [0x01,0xb0]
jal zero, -2048

# CHECK-BYTES: 01 d0
# CHECK-ALIAS: beqz s0, -256
# CHECK-INST: c.beqz s0, -256
# CHECK: # encoding: [0x01,0xd0]
beq s0, zero, -256

# CHECK-BYTES: 7d ec
# CHECk-ALIAS: bnez s0, 254
# CHECK-INST: c.bnez s0, 254
# CHECK: # encoding: [0x7d,0xec]
bne s0, zero, 254

# CHECK-BYTES: 7e 04
# CHECK-ALIAS: slli s0, s0, 31
# CHECK-INST: c.slli s0, 31
# CHECK: # encoding:  [0x7e,0x04]
slli s0, s0, 31

# CHECK-BYTES: fe 50
# CHECK-ALIAS: lw ra, 252(sp)
# CHECK-INST: c.lwsp  ra, 252(sp)
# CHECK: # encoding:  [0xfe,0x50]
lw ra, 252(sp)

# CHECK-BYTES: 82 80
# CHECK-ALIAS: ret
# CHECK-INST: c.jr ra
# CHECK: # encoding:  [0x82,0x80]
jalr zero, 0(ra)

# CHECK-BYTES: 92 80
# CHECK-ALIAS: add ra, zero, tp
# CHECK-INST: c.mv ra, tp
# CHECK:  # encoding: [0x92,0x80]
add ra, zero, tp

# CHECK-BYTES: 92 80
# CHECK-ALIAS: add ra, zero, tp
# CHECK-INST: c.mv ra, tp
# CHECK:  # encoding: [0x92,0x80]
add ra, tp, zero

# CHECK-BYTES: 02 90
# CHECK-ALIAS: ebreak
# CHECK-INST: c.ebreak
# CHECK: # encoding: [0x02,0x90]
ebreak

# CHECK-BYTES: 02 94
# CHECK-ALIAS: jalr s0
# CHECK-INST: c.jalr s0
# CHECK: # encoding: [0x02,0x94]
jalr ra, 0(s0)

# CHECK-BYTES: 3e 94
# CHECK-ALIAS: add s0, s0, a5
# CHECK-INST: c.add s0, a5
# CHECK: # encoding:  [0x3e,0x94]
add s0, a5, s0

# CHECK-BYTES: 3e 94
# CHECK-ALIAS: add s0, s0, a5
# CHECK-INST: c.add s0, a5
# CHECK: # encoding:  [0x3e,0x94]
add s0, s0, a5

# CHECK-BYTES: 82 df
# CHECK-ALIAS: sw zero, 252(sp)
# CHECK-INST: c.swsp zero, 252(sp)
# CHECK: # encoding: [0x82,0xdf]
sw zero, 252(sp)

# CHECK-BYTES: 00 00
# CHECK-ALIAS: unimp
# CHECK-INST: c.unimp
# CHECK: # encoding: [0x00,0x00]
unimp
