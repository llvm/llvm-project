# RUN: llvm-mc -triple=riscv32 -mattr=+c -riscv-no-aliases < %s \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump -d -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-INST %s
# RUN: llvm-mc -triple=riscv32 -mattr=+experimental-zca -riscv-no-aliases < %s \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-zca < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zca -d -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-INST %s

# The following check prefixes are used in this test:
# CHECK-INST.....Match the canonical instr (tests alias to instr. mapping)
# CHECK-EXPAND...Match canonical instr. unconditionally (tests alias expansion)
# CHECK-INST: {{^}}

# CHECK-EXPAND: c.li a0, 0
li x10, 0
# CHECK-EXPAND: c.li a0, 1
li x10, 1
# CHECK-EXPAND: c.li a0, -1
li x10, -1
# CHECK-EXPAND: addi a0, zero, 2047
li x10, 2047
# CHECK-EXPAND: addi a0, zero, -2047
li x10, -2047
# CHECK-EXPAND: c.lui a1, 1
# CHECK-EXPAND: addi a1, a1, -2048
li x11, 2048
# CHECK-EXPAND: addi a1, zero, -2048
li x11, -2048
# CHECK-EXPAND: c.lui a1, 1
# CHECK-EXPAND: addi a1, a1, -2047
li x11, 2049
# CHECK-EXPAND: lui a1, 1048575
# CHECK-EXPAND: addi a1, a1, 2047
li x11, -2049
# CHECK-EXPAND: c.lui a1, 1
# CHECK-EXPAND: c.addi a1, -1
li x11, 4095
# CHECK-EXPAND: lui a1, 1048575
# CHECK-EXPAND: c.addi a1, 1
li x11, -4095
# CHECK-EXPAND: c.lui a2, 1
li x12, 4096
# CHECK-EXPAND: lui a2, 1048575
li x12, -4096
# CHECK-EXPAND: c.lui a2, 1
# CHECK-EXPAND: c.addi a2, 1
li x12, 4097
# CHECK-EXPAND: lui a2, 1048575
# CHECK-EXPAND: c.addi a2, -1
li x12, -4097
# CHECK-EXPAND: lui a2, 524288
# CHECK-EXPAND: c.addi a2, -1
li x12, 2147483647
# CHECK-EXPAND: lui a2, 524288
# CHECK-EXPAND: c.addi a2, 1
li x12, -2147483647
# CHECK-EXPAND: lui a2, 524288
li x12, -2147483648
# CHECK-EXPAND: lui a2, 524288
li x12, -0x80000000

# CHECK-EXPAND: lui a2, 524288
li x12, 0x80000000
# CHECK-EXPAND: c.li a2, -1
li x12, 0xFFFFFFFF

# CHECK-EXPAND: c.mv sp, sp
addi x2, x2, 0
