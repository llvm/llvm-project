# RUN: llvm-mc -triple riscv32 -mattr=+c -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS,CHECK-ALIASASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -show-encoding \
# RUN:   -riscv-no-aliases < %s | FileCheck -check-prefixes=CHECK,CHECK-INST,CHECK-INSTASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS,CHECK-ALIASOBJ32 %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST,CHECK-INSTOBJ32 %s

# RUN: llvm-mc -triple riscv64 -mattr=+c -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ALIAS,CHECK-ALIASASM %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -show-encoding \
# RUN:   -riscv-no-aliases < %s | FileCheck -check-prefixes=CHECK-INST,CHECK-INSTASM %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS,CHECK-ALIASOBJ64 %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST,CHECK-INSTOBJ64 %s

# CHECK-BYTES: 852e
# CHECK-ALIAS: mv a0, a1
# CHECK-INST: c.mv a0, a1
# CHECK: # encoding:  [0x2e,0x85]
addi a0, a1, 0

# CHECK-BYTES: 1fe0
# CHECK-ALIAS: addi s0, sp, 1020
# CHECK-INST: c.addi4spn s0, sp, 1020
# CHECK: # encoding:  [0xe0,0x1f]
addi s0, sp, 1020

# CHECK-BYTES: 5fe0
# CHECK-ALIAS: lw s0, 124(a5)
# CHECK-INST: c.lw s0, 124(a5)
# CHECK: # encoding: [0xe0,0x5f]
lw s0, 124(a5)

# CHECK-BYTES: dfe0
# CHECK-ALIAS: sw s0, 124(a5)
# CHECK-INST: c.sw s0, 124(a5)
# CHECK: # encoding: [0xe0,0xdf]
sw s0, 124(a5)

# CHECK-BYTES: 0001
# CHECK-ALIAS: nop
# CHECK-INST: c.nop
# CHECK: # encoding: [0x01,0x00]
nop

# CHECK-BYTES: 1081
# CHECK-ALIAS: addi ra, ra, -32
# CHECK-INST: c.addi ra, -32
# CHECK: # encoding:  [0x81,0x10]
addi ra, ra, -32

# CHECK-BYTES: 5085
# CHECK-ALIAS: li ra, -31
# CHECK-INST: c.li ra, -31
# CHECK: # encoding: [0x85,0x50]
li ra, -31

# CHECK-BYTES: 7139
# CHECK-ALIAS: addi sp, sp, -64
# CHECK-INST: c.addi16sp sp, -64
# CHECK:  # encoding: [0x39,0x71]
addi sp, sp, -64

# CHECK-BYTES: 61fd
# CHECK-ALIAS: lui gp, 31
# CHECK-INST: c.lui gp, 31
# CHECK: # encoding:  [0xfd,0x61]
lui gp, 31

# CHECK-BYTES: 807d
# CHECK-ALIAS: srli s0, s0, 31
# CHECK-INST: c.srli s0, 31
# CHECK: # encoding:  [0x7d,0x80]
srli s0, s0, 31

# CHECK-BYTES: 847d
# CHECK-ALIAS: srai s0, s0, 31
# CHECK-INST: c.srai s0, 31
# CHECK: # encoding: [0x7d,0x84]
srai s0, s0, 31

# CHECK-BYTES: 887d
# CHECK-ALIAS: andi s0, s0, 31
# CHECK-INST: c.andi s0, 31
# CHECK: # encoding: [0x7d,0x88]
andi s0, s0, 31

# CHECK-BYTES: 8c1d
# CHECK-ALIAS: sub s0, s0, a5
# CHECK-INST: c.sub s0, a5
# CHECK: # encoding: [0x1d,0x8c]
sub s0, s0, a5

# CHECK-BYTES: 8c3d
# CHECK-ALIAS: xor s0, s0, a5
# CHECK-INST: c.xor s0, a5
# CHECK: # encoding: [0x3d,0x8c]
xor s0, s0, a5

# CHECK-BYTES: 8c3d
# CHECK-ALIAS: xor s0, s0, a5
# CHECK-INST: c.xor s0, a5
# CHECK: # encoding: [0x3d,0x8c]
xor s0, a5, s0

# CHECK-BYTES: 8c5d
# CHECK-ALIAS: or s0, s0, a5
# CHECK-INST: c.or s0, a5
# CHECK: # encoding:  [0x5d,0x8c]
or s0, s0, a5

# CHECK-BYTES: 8c45
# CHECK-ALIAS: or s0, s0, s1
# CHECK-INST: c.or s0, s1
# CHECK:  # encoding: [0x45,0x8c]
or  s0, s1, s0

# CHECK-BYTES: 8c7d
# CHECK-ALIAS: and s0, s0, a5
# CHECK-INST: c.and s0, a5
# CHECK: # encoding: [0x7d,0x8c]
and s0, s0, a5

# CHECK-BYTES: 8c7d
# CHECK-ALIAS: and s0, s0, a5
# CHECK-INST: c.and s0, a5
# CHECK: # encoding: [0x7d,0x8c]
and s0, a5, s0

# CHECK-BYTES: b001
# CHECK-ALIASASM: j -2048
# CHECK-ALIASOBJ32: j 0xfffff826
# CHECK-ALIASOBJ64: j 0xfffffffffffff826
# CHECK-INSTASM: c.j -2048
# CHECK-INSTOBJ32: c.j 0xfffff826
# CHECK-INSTOBJ64: c.j 0xfffffffffffff826
# CHECK:  # encoding: [0x01,0xb0]
jal zero, -2048

# CHECK-BYTES: d001
# CHECK-ALIASASM: beqz s0, -256
# CHECK-ALIASOBJ32: beqz s0, 0xffffff28
# CHECK-ALIASOBJ64: beqz s0, 0xffffffffffffff28
# CHECK-INSTASM: c.beqz s0, -256
# CHECK-INSTOBJ32: c.beqz s0, 0xffffff28
# CHECK-INSTOBJ64: c.beqz s0, 0xffffffffffffff28
# CHECK: # encoding: [0x01,0xd0]
beq s0, zero, -256

# CHECK-BYTES: d001
# CHECK-ALIASASM: beqz s0, -256
# CHECK-ALIASOBJ32: beqz s0, 0xffffff2a
# CHECK-ALIASOBJ64: beqz s0, 0xffffffffffffff2a
# CHECK-INSTASM: c.beqz s0, -256
# CHECK-INSTOBJ32: c.beqz s0, 0xffffff2a
# CHECK-INSTOBJ64: c.beqz s0, 0xffffffffffffff2a
# CHECK: # encoding: [0x01,0xd0]
beq zero, s0, -256

# CHECK-BYTES: ec7d
# CHECK-ALIASASM: bnez s0, 254
# CHECK-ALIASOBJ32: bnez s0, 0x12a
# CHECK-ALIASOBJ64: bnez s0, 0x12a
# CHECK-INSTASM: c.bnez s0, 254
# CHECK-INSTOBJ32: c.bnez s0, 0x12a
# CHECK-INSTOBJ64: c.bnez s0, 0x12a
# CHECK: # encoding: [0x7d,0xec]
bne s0, zero, 254

# CHECK-BYTES: ec7d
# CHECK-ALIASASM: bnez s0, 254
# CHECK-ALIASOBJ32: bnez s0, 0x12c
# CHECK-ALIASOBJ64: bnez s0, 0x12c
# CHECK-INSTASM: c.bnez s0, 254
# CHECK-INSTOBJ32: c.bnez s0, 0x12c
# CHECK-INSTOBJ64: c.bnez s0, 0x12c
# CHECK: # encoding: [0x7d,0xec]
bne zero, s0, 254

# CHECK-BYTES: 047e
# CHECK-ALIAS: slli s0, s0, 31
# CHECK-INST: c.slli s0, 31
# CHECK: # encoding:  [0x7e,0x04]
slli s0, s0, 31

# CHECK-BYTES: 50fe
# CHECK-ALIAS: lw ra, 252(sp)
# CHECK-INST: c.lwsp  ra, 252(sp)
# CHECK: # encoding:  [0xfe,0x50]
lw ra, 252(sp)

# CHECK-BYTES: 8082
# CHECK-ALIAS: ret
# CHECK-INST: c.jr ra
# CHECK: # encoding:  [0x82,0x80]
jalr zero, 0(ra)

# CHECK-BYTES: 8092
# CHECK-ALIAS: mv ra, tp
# CHECK-INST: c.mv ra, tp
# CHECK:  # encoding: [0x92,0x80]
add ra, zero, tp

# CHECK-BYTES: 8092
# CHECK-ALIAS: mv ra, tp
# CHECK-INST: c.mv ra, tp
# CHECK:  # encoding: [0x92,0x80]
add ra, tp, zero

# CHECK-BYTES: 9002
# CHECK-ALIAS: ebreak
# CHECK-INST: c.ebreak
# CHECK: # encoding: [0x02,0x90]
ebreak

# CHECK-BYTES: 9402
# CHECK-ALIAS: jalr s0
# CHECK-INST: c.jalr s0
# CHECK: # encoding: [0x02,0x94]
jalr ra, 0(s0)

# CHECK-BYTES: 943e
# CHECK-ALIAS: add s0, s0, a5
# CHECK-INST: c.add s0, a5
# CHECK: # encoding:  [0x3e,0x94]
add s0, a5, s0

# CHECK-BYTES: 943e
# CHECK-ALIAS: add s0, s0, a5
# CHECK-INST: c.add s0, a5
# CHECK: # encoding:  [0x3e,0x94]
add s0, s0, a5

# CHECK-BYTES: df82
# CHECK-ALIAS: sw zero, 252(sp)
# CHECK-INST: c.swsp zero, 252(sp)
# CHECK: # encoding: [0x82,0xdf]
sw zero, 252(sp)

# CHECK-BYTES: 0000
# CHECK-ALIAS: unimp
# CHECK-INST: c.unimp
# CHECK: # encoding: [0x00,0x00]
unimp
