# RUN: not llvm-mc -triple=riscv32 < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK,CHECK-RV32,CHECK-RV32I %s
# RUN: not llvm-mc -triple=riscv64 < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK,CHECK-RV64,CHECK-RV64I %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+f < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK,CHECK-RV32,CHECK-RV32IF %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+f < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK,CHECK-RV64,CHECK-RV64IF %s

# Tests for the mnemonic spell checker. Suggestions should only include those
# which are valid for the current set of features

ad x1, x1, x1
# CHECK-RV32: did you mean: add, addi, and, andi, la
# CHECK-RV64: did you mean: add, addi, addw, and, andi, la, ld, sd
# CHECK-NEXT: ad x1, x1, x1

fl ft0, 0(sp)
# CHECK-RV32I: did you mean: la, lb, lh, li, lw
# CHECK-RV32IF: did you mean: flw, la, lb, lh, li, lw
# CHECK-RV64I: did you mean: la, lb, ld, lh, li, lw
# CHECK-RV64IF: did you mean: flw, la, lb, ld, lh, li, lw
# CHECK-NEXT: fl ft0, 0(sp)

addd x1, x1, x1
# CHECK-RV32: did you mean: add, addi
# CHECK-RV64: did you mean: add, addi, addw
# CHECK-NEXT: addd x1, x1, x1

vm x0, x0
# CHECK: did you mean: mv
# CHECK-NEXT: vm x0, x0
