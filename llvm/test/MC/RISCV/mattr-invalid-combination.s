# RUN: not --crash llvm-mc -triple riscv64 -mattr=+e < %s 2>&1 \
# RUN:   | FileCheck %s -check-prefix=RV64E

# RV64E: LLVM ERROR: standard user-level extension 'e' requires 'rv32'
