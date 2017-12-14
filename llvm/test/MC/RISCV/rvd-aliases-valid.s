# RUN: llvm-mc %s -triple=riscv32 -mattr=+d -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+d -riscv-no-aliases=false \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+d -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+d -riscv-no-aliases=false \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+d < %s \
# RUN:     | llvm-objdump -d -mattr=+d -riscv-no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+d < %s \
# RUN:     | llvm-objdump -d -mattr=+d -riscv-no-aliases=false - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d < %s \
# RUN:     | llvm-objdump -d -mattr=+d -riscv-no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d < %s \
# RUN:     | llvm-objdump -d -mattr=+d -riscv-no-aliases=false - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

# TODO fld
# TODO fsd

# CHECK-INST: fsgnj.d ft0, ft1, ft1
# CHECK-ALIAS: fmv.d ft0, ft1
fmv.d f0, f1
# CHECK-INST: fsgnjx.d ft1, ft2, ft2
# CHECK-ALIAS: fabs.d ft1, ft2
fabs.d f1, f2
# CHECK-INST: fsgnjn.d ft2, ft3, ft3
# CHECK-ALIAS: fneg.d ft2, ft3
fneg.d f2, f3
