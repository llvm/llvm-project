# RUN: not llvm-mc -triple riscv64 -mattr=+e < %s 2>&1 \
# RUN:   | FileCheck %s -check-prefix=RV64E

# RV64E: LLVM ERROR: RV32E can't be enabled for an RV64 target
