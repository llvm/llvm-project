# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-INST,CHECK-ASM-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-ALIAS,CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-EXPAND,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-ALIAS %s

# The following check prefixes are used in this test:
# CHECK-INST.....Match the canonical instr (tests alias to instr. mapping)
# CHECK-ALIAS....Match the alias (tests instr. to alias mapping)
# CHECK-EXPAND...Match canonical instr. unconditionally (tests alias expansion)

# Needed for testing valid %pcrel_lo expressions
.Lpcrel_hi0: auipc a0, %pcrel_hi(foo)

# Needed for testing li with a symbol difference
.Lbuf: .skip 8
.Lbuf_end:

# CHECK-INST: addi a0, zero, 0
# CHECK-ALIAS: li a0, 0
li x10, 0
# CHECK-INST: addi a0, zero, 1
# CHECK-ALIAS: li a0, 1
li x10, 1
# CHECK-INST: addi a0, zero, -1
# CHECK-ALIAS: li a0, -1
li x10, -1
# CHECK-INST: addi a0, zero, 2047
# CHECK-ALIAS: li a0, 2047
li x10, 2047
# CHECK-INST: addi a0, zero, -2047
# CHECK-ALIAS: li a0, -2047
li x10, -2047
# CHECK-INST: addi a1, zero, 1
# CHECK-INST: slli a1, a1, 11
# CHECK-ALIAS: li a1, 1
# CHECK-ALIAS: slli a1, a1, 11
li x11, 2048
# CHECK-INST: addi a1, zero, -2048
# CHECK-ALIAS: li a1, -2048
li x11, -2048
# CHECK-EXPAND: lui a1, 1
# CHECK-EXPAND: addi a1, a1, -2047
li x11, 2049
# CHECK-EXPAND: lui a1, 1048575
# CHECK-EXPAND: addi a1, a1, 2047
li x11, -2049
# CHECK-EXPAND: lui a1, 1
# CHECK-EXPAND: addi a1, a1, -1
li x11, 4095
# CHECK-EXPAND: lui a1, 1048575
# CHECK-EXPAND: addi a1, a1, 1
li x11, -4095
# CHECK-EXPAND: lui a2, 1
li x12, 4096
# CHECK-EXPAND: lui a2, 1048575
li x12, -4096
# CHECK-EXPAND: lui a2, 1
# CHECK-EXPAND: addi a2, a2, 1
li x12, 4097
# CHECK-EXPAND: lui a2, 1048575
# CHECK-EXPAND: addi a2, a2, -1
li x12, -4097
# CHECK-EXPAND: lui a2, 524288
# CHECK-EXPAND: addi a2, a2, -1
li x12, 2147483647
# CHECK-EXPAND: lui a2, 524288
# CHECK-EXPAND: addi a2, a2, 1
li x12, -2147483647
# CHECK-EXPAND: lui a2, 524288
li x12, -2147483648
# CHECK-EXPAND: lui a2, 524288
li x12, -0x80000000

# CHECK-EXPAND: lui a2, 524288
li x12, 0x80000000
# CHECK-INST: addi a2, zero, -1
# CHECK-ALIAS: li a2, -1
li x12, 0xFFFFFFFF

# CHECK-INST: addi a0, zero, 1110
# CHECK-ALIAS: li a0, 1110
li a0, %lo(0x123456)

# CHECK-OBJ-NOALIAS: addi a0, zero, 0
# CHECK-OBJ: R_RISCV_LO12
li a0, %lo(foo)
# CHECK-OBJ-NOALIAS: addi a0, zero, 0
# CHECK-OBJ: R_RISCV_PCREL_LO12
li a0, %pcrel_lo(.Lpcrel_hi0)

.equ CONST, 0x123456
# CHECK-EXPAND: lui a0, 291
# CHECK-EXPAND: addi a0, a0, 1110
li a0, CONST
# CHECK-EXPAND: lui a0, 291
# CHECK-EXPAND: addi a0, a0, 1111
li a0, CONST+1

.equ CONST, 0x654321
# CHECK-EXPAND: lui a0, 1620
# CHECK-EXPAND: addi a0, a0, 801
li a0, CONST

.equ CONST, .Lbuf_end - .Lbuf
# CHECK-ASM: li a0, CONST
# CHECK-ASM-NOALIAS: addi a0, zero, CONST
# CHECK-OBJ-NOALIAS: addi a0, zero, 8
li a0, CONST

# CHECK-ASM: addi a0, zero, .Lbuf_end-.Lbuf
# CHECK-ASM-NOALIAS: addi a0, zero, .Lbuf_end-.Lbuf
# CHECK-OBJ-NOALIAS: addi a0, zero, 8
li a0, .Lbuf_end - .Lbuf

# CHECK-INST: addi a0, zero, 0
# CHECK-ALIAS: li a0, 0
la x10, 0
lla x10, 0
# CHECK-INST: addi a0, zero, 1
# CHECK-ALIAS: li a0, 1
la x10, 1
lla x10, 1
# CHECK-INST: addi a0, zero, -1
# CHECK-ALIAS: li a0, -1
la x10, -1
lla x10, -1
# CHECK-INST: addi a0, zero, 2047
# CHECK-ALIAS: li a0, 2047
la x10, 2047
lla x10, 2047
# CHECK-INST: addi a0, zero, -2047
# CHECK-ALIAS: li a0, -2047
la x10, -2047
lla x10, -2047
# CHECK-INST: addi a1, zero, 1
# CHECK-INST: slli a1, a1, 11
# CHECK-ALIAS: li a1, 1
# CHECK-ALIAS: slli a1, a1, 11
la x11, 2048
lla x11, 2048
# CHECK-INST: addi a1, zero, -2048
# CHECK-ALIAS: li a1, -2048
la x11, -2048
lla x11, -2048
# CHECK-EXPAND: lui a1, 1
# CHECK-EXPAND: addi a1, a1, -2047
la x11, 2049
lla x11, 2049
# CHECK-EXPAND: lui a1, 1048575
# CHECK-EXPAND: addi a1, a1, 2047
la x11, -2049
lla x11, -2049
# CHECK-EXPAND: lui a1, 1
# CHECK-EXPAND: addi a1, a1, -1
la x11, 4095
lla x11, 4095
# CHECK-EXPAND: lui a1, 1048575
# CHECK-EXPAND: addi a1, a1, 1
la x11, -4095
lla x11, -4095
# CHECK-EXPAND: lui a2, 1
la x12, 4096
lla x12, 4096
# CHECK-EXPAND: lui a2, 1048575
la x12, -4096
lla x12, -4096
# CHECK-EXPAND: lui a2, 1
# CHECK-EXPAND: addi a2, a2, 1
la x12, 4097
lla x12, 4097
# CHECK-EXPAND: lui a2, 1048575
# CHECK-EXPAND: addi a2, a2, -1
la x12, -4097
lla x12, -4097
# CHECK-EXPAND: lui a2, 524288
# CHECK-EXPAND: addi a2, a2, -1
la x12, 2147483647
lla x12, 2147483647
# CHECK-EXPAND: lui a2, 524288
# CHECK-EXPAND: addi a2, a2, 1
la x12, -2147483647
lla x12, -2147483647
# CHECK-EXPAND: lui a2, 524288
la x12, -2147483648
lla x12, -2147483648
# CHECK-EXPAND: lui a2, 524288
la x12, -0x80000000
lla x12, -0x80000000

# CHECK-EXPAND: lui a2, 524288
la x12, 0x80000000
lla x12, 0x80000000
# CHECK-INST: addi a2, zero, -1
# CHECK-ALIAS: li a2, -1
la x12, 0xFFFFFFFF
lla x12, 0xFFFFFFFF

.equ CONSTANT, 0x123456
# CHECK-EXPAND: lui a0, 291
# CHECK-EXPAND: addi a0, a0, 1110
la a0, CONSTANT
lla a0, CONSTANT
# CHECK-EXPAND: lui a0, 291
# CHECK-EXPAND: addi a0, a0, 1111
la a0, CONSTANT+1
lla a0, CONSTANT+1

.equ CONSTANT, 0x654321
# CHECK-EXPAND: lui a0, 1620
# CHECK-EXPAND: addi a0, a0, 801
la a0, CONSTANT
lla a0, CONSTANT

# CHECK-INST: csrrs t4, instreth, zero
# CHECK-ALIAS: rdinstreth t4
rdinstreth x29
# CHECK-INST: csrrs s11, cycleh, zero
# CHECK-ALIAS: rdcycleh s11
rdcycleh x27
# CHECK-INST: csrrs t3, timeh, zero
# CHECK-ALIAS: rdtimeh t3
rdtimeh x28

# CHECK-EXPAND: lb a0, 0(a1)
lb x10, (x11)
# CHECK-EXPAND: lh a0, 0(a1)
lh x10, (x11)
# CHECK-EXPAND: lw a0, 0(a1)
lw x10, (x11)
# CHECK-EXPAND: lbu a0, 0(a1)
lbu x10, (x11)
# CHECK-EXPAND: lhu a0, 0(a1)
lhu x10, (x11)

# CHECK-EXPAND: sb a0, 0(a1)
sb x10, (x11)
# CHECK-EXPAND: sh a0, 0(a1)
sh x10, (x11)
# CHECK-EXPAND: sw a0, 0(a1)
sw x10, (x11)

# CHECK-EXPAND: slli a0, a1, 24
# CHECK-EXPAND: srai a0, a0, 24
sext.b x10, x11

# CHECK-EXPAND: slli a0, a1, 16
# CHECK-EXPAND: srai a0, a0, 16
sext.h x10, x11

# CHECK-INST: andi a0, a1, 255
# CHECK-ALIAS: andi a0, a1, 255
zext.b x10, x11

# CHECK-EXPAND: slli a0, a1, 16
# CHECK-EXPAND: srli a0, a0, 16
zext.h x10, x11
