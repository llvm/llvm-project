# RUN: llvm-mc %s -triple=riscv32 -mattr=+q -M no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+q \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+q -M no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+q \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+q < %s \
# RUN:     | llvm-objdump -d --mattr=+q --no-print-imm-hex -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+q < %s \
# RUN:     | llvm-objdump -d --mattr=+q --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+q < %s \
# RUN:     | llvm-objdump -d --mattr=+q --no-print-imm-hex -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+q < %s \
# RUN:     | llvm-objdump -d --mattr=+q --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
##===----------------------------------------------------------------------===##

# CHECK-INST: flq ft0, 0(a0)
# CHECK-ALIAS:  flq ft0, 0(a0)
flq f0, (a0)
# CHECK-INST: fsq ft0, 0(a0)
# CHECK-ALIAS: fsq ft0, 0(a0)
fsq f0, (a0)

# CHECK-INST: fsgnj.q ft0, ft1, ft1
# CHECK-ALIAS: fmv.q ft0, ft1
fmv.q f0, f1
# CHECK-INST: fsgnjx.q ft1, ft2, ft2
# CHECK-ALIAS: fabs.q ft1, ft2
fabs.q f1, f2
# CHECK-INST: fsgnjn.q ft2, ft3, ft3
# CHECK-ALIAS: fneg.q ft2, ft3
fneg.q f2, f3

# CHECK-INST: flt.q tp, ft6, ft5
# CHECK-ALIAS: flt.q tp, ft6, ft5
fgt.q x4, f5, f6
# CHECK-INST: fle.q t2, fs1, fs0
# CHECK-ALIAS: fle.q t2, fs1, fs0
fge.q x7, f8, f9

# CHECK-INST: flq ft0, 0(a0)
# CHECK-ALIAS: flq ft0, 0(a0)
flq f0, (x10)
# CHECK-INST: fsq ft0, 0(a0)
# CHECK-ALIAS: fsq ft0, 0(a0)
fsq f0, (x10)
