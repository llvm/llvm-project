# Xqcili - Check aliases for li instruction
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcili -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-INST-RISCV32 %s

# CHECK-INST: qc.li  a0, 2048
# CHECK-INST-RISCV32: addi    a0, zero, 1
# CHECK-INST-RISCV32: slli    a0, a0, 11
li x10, 2048

# CHECK-INST: addi  a0, zero, 2047
# CHECK-INST-RISCV32: addi    a0, zero, 2047
li x10, 2047

# CHECK-INST: addi  a0, zero, -2048
# CHECK-INST-RISCV32: addi    a0, zero, -2048
li x10, -2048

# CHECK-INST: addi  a0, zero, -2047
# CHECK-INST-RISCV32: addi    a0, zero, -2047
li x10, -2047

# CHECK-INST: lui   a0, 512
# CHECK-INST-RISCV32: lui     a0, 512
li x10, 2097152

# CHECK-INST: lui   a0, 1
# CHECK-INST-RISCV32: lui     a0, 1
li x10, 4096

# CHECK-INST: qc.e.li  a0, 1048577
# CHECK-INST-RISCV32: lui     a0, 256
# CHECK-INST-RISCV32: addi    a0, a0, 1
li x10, 1048577

# CHECK-INST: lui   a0, 512
# CHECK-INST-RISCV32: lui     a0, 512
li x10, 2097152
