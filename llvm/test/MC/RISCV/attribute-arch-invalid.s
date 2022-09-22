## Invalid arch string

# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

## Version strings are required for experimental extensions

.attribute arch, "rv32izbe"
# CHECK:  error: invalid arch name 'rv32izbe', experimental extension requires explicit version number `zbe`

.attribute arch, "rv32izbf"
# CHECK: error: invalid arch name 'rv32izbf', experimental extension requires explicit version number `zbf`

.attribute arch, "rv32izbm"
# CHECK: error: invalid arch name 'rv32izbm', experimental extension requires explicit version number `zbm`

.attribute arch, "rv32izbr"
# CHECK: error: invalid arch name 'rv32izbr', experimental extension requires explicit version number `zbr`

.attribute arch, "rv32izvfh"
# CHECK: error: invalid arch name 'rv32izvfh', experimental extension requires explicit version number `zvfh`
