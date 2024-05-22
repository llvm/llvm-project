# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zicfilp -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zicfilp -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-NO-EXT: immediate must be an integer in the range [0, 1048575]
lpad 1048576
