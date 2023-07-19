## Invalid arch string

# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

## Version strings are required for experimental extensions

.attribute arch, "rv32izvfbfmin"
# CHECK: error: invalid arch name 'rv32izvfbfmin', experimental extension requires explicit version number `zvfbfmin`
