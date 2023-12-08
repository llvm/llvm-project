# RUN: llvm-mc %s -triple=riscv32 -mattr=+zdinx -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zdinx \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zdinx -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zdinx \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zdinx %s \
# RUN:     | llvm-objdump -d --mattr=+zdinx -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zdinx %s \
# RUN:     | llvm-objdump -d --mattr=+zdinx - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zdinx %s \
# RUN:     | llvm-objdump -d --mattr=+zdinx -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zdinx %s \
# RUN:     | llvm-objdump -d --mattr=+zdinx - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Aliases which omit the rounding mode.
##===----------------------------------------------------------------------===##

# CHECK-INST: fmadd.d a0, a2, a4, a6, dyn
# CHECK-ALIAS: fmadd.d a0, a2, a4, a6
fmadd.d x10, x12, x14, x16
# CHECK-INST: fmsub.d a0, a2, a4, a6, dyn
# CHECK-ALIAS: fmsub.d a0, a2, a4, a6
fmsub.d x10, x12, x14, x16
# CHECK-INST: fnmsub.d a0, a2, a4, a6, dyn
# CHECK-ALIAS: fnmsub.d a0, a2, a4, a6
fnmsub.d x10, x12, x14, x16
# CHECK-INST: fnmadd.d a0, a2, a4, a6, dyn
# CHECK-ALIAS: fnmadd.d a0, a2, a4, a6
fnmadd.d x10, x12, x14, x16
# CHECK-INST: fadd.d a0, a2, a4, dyn
# CHECK-ALIAS: fadd.d a0, a2, a4
fadd.d x10, x12, x14
# CHECK-INST: fsub.d a0, a2, a4, dyn
# CHECK-ALIAS: fsub.d a0, a2, a4
fsub.d x10, x12, x14
# CHECK-INST: fmul.d a0, a2, a4, dyn
# CHECK-ALIAS: fmul.d a0, a2, a4
fmul.d x10, x12, x14
# CHECK-INST: fdiv.d a0, a2, a4, dyn
# CHECK-ALIAS: fdiv.d a0, a2, a4
fdiv.d x10, x12, x14
