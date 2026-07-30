# RUN: llvm-mc %s -triple=riscv32 -filetype=asm -riscv-add-build-attributes \
# RUN:   | FileCheck %s --check-prefixes=RV32
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm -riscv-add-build-attributes \
# RUN:   | FileCheck %s --check-prefixes=RV64
# RUN: llvm-mc %s -triple=riscv32 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+m | FileCheck %s --check-prefixes=RV32M
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+m | FileCheck %s --check-prefixes=RV64M

# RV32-NOT: attribute 4
# RV32: attribute 5, "rv32i2p1"

# RV64-NOT: attribute 4
# RV64: attribute 5, "rv64i2p1"

# RV32M-NOT: attribute 4
# RV32M: attribute 5, "rv32i2p1_m2p0_zmmul1p0"

# RV64M-NOT: attribute 4
# RV64M: attribute 5, "rv64i2p1_m2p0_zmmul1p0"
