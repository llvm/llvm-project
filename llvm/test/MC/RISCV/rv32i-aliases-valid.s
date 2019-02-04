# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -riscv-no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-EXPAND,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-ALIAS %s

# The following check prefixes are used in this test:
# CHECK-INST.....Match the canonical instr (tests alias to instr. mapping)
# CHECK-ALIAS....Match the alias (tests instr. to alias mapping)
# CHECK-EXPAND...Match canonical instr. unconditionally (tests alias expansion)


# CHECK-INST: addi a0, zero, 0
# CHECK-ALIAS: mv a0, zero
li x10, 0
# CHECK-EXPAND: addi a0, zero, 1
li x10, 1
# CHECK-EXPAND: addi a0, zero, -1
li x10, -1
# CHECK-EXPAND: addi a0, zero, 2047
li x10, 2047
# CHECK-EXPAND: addi a0, zero, -2047
li x10, -2047
# CHECK-EXPAND: lui a1, 1
# CHECK-EXPAND: addi a1, a1, -2048
li x11, 2048
# CHECK-EXPAND: addi a1, zero, -2048
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
# CHECK-EXPAND: addi a2, zero, -1
li x12, 0xFFFFFFFF

# CHECK-EXPAND: addi a0, zero, 1110
li a0, %lo(0x123456)
# CHECK-OBJ-NOALIAS: addi a0, zero, 0
# CHECK-OBJ: R_RISCV_PCREL_LO12
li a0, %pcrel_lo(0x123456)

# CHECK-OBJ-NOALIAS: addi a0, zero, 0
# CHECK-OBJ: R_RISCV_LO12
li a0, %lo(foo)
# CHECK-OBJ-NOALIAS: addi a0, zero, 0
# CHECK-OBJ: R_RISCV_PCREL_LO12
li a0, %pcrel_lo(foo)

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

# CHECK-INST: csrrs t4, instreth, zero
# CHECK-ALIAS: rdinstreth t4
rdinstreth x29
# CHECK-INST: csrrs s11, cycleh, zero
# CHECK-ALIAS: rdcycleh s11
rdcycleh x27
# CHECK-INST: csrrs t3, timeh, zero
# CHECK-ALIAS: rdtimeh t3
rdtimeh x28
