# RUN: not llvm-mc -triple=riscv64 -mattr=+experimental-y,+zce /dev/null 2>&1 | FileCheck %s --check-prefix=RV64Y-ZCE
# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-y,+zcf /dev/null 2>&1 | FileCheck %s --check-prefix=RV32Y-ZCF
# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-y,+zclsd /dev/null 2>&1 | FileCheck %s --check-prefix=RV32Y-ZCLSD
# RUN: not llvm-mc -triple=riscv64 -mattr=+experimental-y,+zcd /dev/null 2>&1 | FileCheck %s --check-prefix=RV64Y-ZCD
# RUN: not llvm-mc -triple=riscv64 -mattr=+experimental-y,+c,+zcmp /dev/null 2>&1 | FileCheck %s --check-prefix=RV64Y-C-ZCMP
# RUN: not llvm-mc -triple=riscv64 -mattr=+experimental-y,+zca,+zcmp /dev/null 2>&1 | FileCheck %s --check-prefix=RV64Y-ZCA-ZCMP

# RUN: not llvm-mc -triple=riscv64 < %s 2>&1 | FileCheck %s --check-prefix=ATTR --implicit-check-not="error:"

# RV64Y-ZCE: LLVM ERROR: 'zcmt' is incompatible with rv64y base
# RV32Y-ZCF: LLVM ERROR: 'zcf' is incompatible with rv32y base
# RV32Y-ZCLSD: LLVM ERROR: 'zclsd' is incompatible with rv32y base
# RV64Y-ZCD: LLVM ERROR: 'zcd' is incompatible with rv64y base
# RV64Y-C-ZCMP: LLVM ERROR: 'zcmp' is incompatible with rv64y base
# RV64Y-ZCA-ZCMP: LLVM ERROR: 'zcmp' is incompatible with rv64y base

.attribute arch, "rv64y0p98_zce"
# ATTR: error: invalid arch name 'rv64y0p98_zce', 'zcmt' is incompatible with rv64y base

.attribute arch, "rv32y0p98_zcf"
# ATTR: error: invalid arch name 'rv32y0p98_zcf', 'zcf' is incompatible with rv32y base

.attribute arch, "rv32y0p98_zclsd"
# ATTR: error: invalid arch name 'rv32y0p98_zclsd', 'zclsd' is incompatible with rv32y base

.attribute arch, "rv64y0p98_zcd"
# ATTR: error: invalid arch name 'rv64y0p98_zcd', 'zcd' is incompatible with rv64y base

.attribute arch, "rv64y0p98_c_zcmp"
# ATTR: error: invalid arch name 'rv64y0p98_c_zcmp', 'zcmp' is incompatible with rv64y base

.attribute arch, "rv64y0p98_zca_zcmp"
# ATTR: error: invalid arch name 'rv64y0p98_zca_zcmp', 'zcmp' is incompatible with rv64y base