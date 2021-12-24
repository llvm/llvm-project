## Invalid arch string

# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

## Version strings are required for experimental extensions

.attribute arch, "rv32iv"
# CHECK: error: invalid arch name 'rv32iv', experimental extension requires explicit version number `v`

.attribute arch, "rv32izba"
# CHECK: error: invalid arch name 'rv32izba', experimental extension requires explicit version number `zba`

.attribute arch, "rv32izbb"
# CHECK: error: invalid arch name 'rv32izbb', experimental extension requires explicit version number `zbb`

.attribute arch, "rv32izbc"
# CHECK: error: invalid arch name 'rv32izbc', experimental extension requires explicit version number `zbc`

.attribute arch, "rv32izbe"
# CHECK:  error: invalid arch name 'rv32izbe', experimental extension requires explicit version number `zbe`

.attribute arch, "rv32izbf"
# CHECK: error: invalid arch name 'rv32izbf', experimental extension requires explicit version number `zbf`

.attribute arch, "rv32izbm"
# CHECK: error: invalid arch name 'rv32izbm', experimental extension requires explicit version number `zbm`

.attribute arch, "rv32izbp"
# CHECK: error: invalid arch name 'rv32izbp', experimental extension requires explicit version number `zbp`

.attribute arch, "rv32izbr"
# CHECK: error: invalid arch name 'rv32izbr', experimental extension requires explicit version number `zbr`

.attribute arch, "rv32izbs"
# CHECK: error: invalid arch name 'rv32izbs', experimental extension requires explicit version number `zbs`

.attribute arch, "rv32izbt"
# CHECK: error: invalid arch name 'rv32izbt', experimental extension requires explicit version number `zbt`

.attribute arch, "rv32ifzfhmin"
# CHECK: error: invalid arch name 'rv32ifzfhmin', experimental extension requires explicit version number `zfhmin`

.attribute arch, "rv32ifzfh"
# CHECK: error: invalid arch name 'rv32ifzfh', experimental extension requires explicit version number `zfh`

.attribute arch, "rv32ivzvlsseg"
# CHECK: error: invalid arch name 'rv32ivzvlsseg', experimental extension requires explicit version number `v`
