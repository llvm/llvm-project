# RUN: llvm-mc %s -triple=riscv64 -mattr=+d -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+d \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d < %s \
# RUN:     | llvm-objdump -d -mattr=+d -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d < %s \
# RUN:     | llvm-objdump -d -mattr=+d - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Aliases which omit the rounding mode.
##===----------------------------------------------------------------------===##

# CHECK-INST: fcvt.l.d a0, ft0, dyn
# CHECK-ALIAS: fcvt.l.d a0, ft0{{[[:space:]]}}
fcvt.l.d a0, ft0
# CHECK-INST: fcvt.lu.d a1, ft1, dyn
# CHECK-ALIAS: fcvt.lu.d a1, ft1{{[[:space:]]}}
fcvt.lu.d a1, ft1
# CHECK-INST: fcvt.d.l ft3, a3, dyn
# CHECK-ALIAS: fcvt.d.l ft3, a3{{[[:space:]]}}
fcvt.d.l ft3, a3
# CHECK-INST: fcvt.d.lu ft4, a4, dyn
# CHECK-ALIAS: fcvt.d.lu ft4, a4{{[[:space:]]}}
fcvt.d.lu ft4, a4
