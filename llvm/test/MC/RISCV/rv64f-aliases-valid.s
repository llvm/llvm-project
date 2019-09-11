# RUN: llvm-mc %s -triple=riscv64 -mattr=+f -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+f \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d -mattr=+f -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d -mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Aliases which omit the rounding mode.
##===----------------------------------------------------------------------===##

# CHECK-INST: fcvt.l.s a0, ft0, dyn
# CHECK-ALIAS: fcvt.l.s a0, ft0{{[[:space:]]}}
fcvt.l.s a0, ft0
# CHECK-INST: fcvt.lu.s a1, ft1, dyn
# CHECK-ALIAS: fcvt.lu.s a1, ft1{{[[:space:]]}}
fcvt.lu.s a1, ft1
# CHECK-INST: fcvt.s.l ft2, a2, dyn
# CHECK-ALIAS: fcvt.s.l ft2, a2{{[[:space:]]}}
fcvt.s.l ft2, a2
# CHECK-INST: fcvt.s.lu ft3, a3, dyn
# CHECK-ALIAS: fcvt.s.lu ft3, a3{{[[:space:]]}}
fcvt.s.lu ft3, a3
