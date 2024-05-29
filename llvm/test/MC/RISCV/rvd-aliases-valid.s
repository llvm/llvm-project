# RUN: llvm-mc %s -triple=riscv32 -mattr=+d -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+d \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+d -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+d \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+d < %s \
# RUN:     | llvm-objdump -d --mattr=+d --no-print-imm-hex -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+d < %s \
# RUN:     | llvm-objdump -d --mattr=+d --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d < %s \
# RUN:     | llvm-objdump -d --mattr=+d --no-print-imm-hex -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d < %s \
# RUN:     | llvm-objdump -d --mattr=+d --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
##===----------------------------------------------------------------------===##

# CHECK-INST: fld ft0, 0(a0)
# CHECK-ALIAS:  fld ft0, 0(a0)
fld f0, (a0)
# CHECK-INST: fsd ft0, 0(a0)
# CHECK-ALIAS: fsd ft0, 0(a0)
fsd f0, (a0)

# CHECK-INST: fsgnj.d ft0, ft1, ft1
# CHECK-ALIAS: fmv.d ft0, ft1
fmv.d f0, f1
# CHECK-INST: fsgnjx.d ft1, ft2, ft2
# CHECK-ALIAS: fabs.d ft1, ft2
fabs.d f1, f2
# CHECK-INST: fsgnjn.d ft2, ft3, ft3
# CHECK-ALIAS: fneg.d ft2, ft3
fneg.d f2, f3

# CHECK-INST: flt.d tp, ft6, ft5
# CHECK-ALIAS: flt.d tp, ft6, ft5
fgt.d x4, f5, f6
# CHECK-INST: fle.d t2, fs1, fs0
# CHECK-ALIAS: fle.d t2, fs1, fs0
fge.d x7, f8, f9

# CHECK-INST: fld ft0, 0(a0)
# CHECK-ALIAS: fld ft0, 0(a0)
fld f0, (x10)
# CHECK-INST: fsd ft0, 0(a0)
# CHECK-ALIAS: fsd ft0, 0(a0)
fsd f0, (x10)

##===----------------------------------------------------------------------===##
## Aliases which omit the rounding mode.
##===----------------------------------------------------------------------===##

# CHECK-INST: fmadd.d fa0, fa1, fa2, fa3, dyn
# CHECK-ALIAS: fmadd.d fa0, fa1, fa2, fa3{{[[:space:]]}}
fmadd.d f10, f11, f12, f13
# CHECK-INST: fmsub.d fa4, fa5, fa6, fa7, dyn
# CHECK-ALIAS: fmsub.d fa4, fa5, fa6, fa7{{[[:space:]]}}
fmsub.d f14, f15, f16, f17
# CHECK-INST: fnmsub.d fs2, fs3, fs4, fs5, dyn
# CHECK-ALIAS: fnmsub.d fs2, fs3, fs4, fs5{{[[:space:]]}}
fnmsub.d f18, f19, f20, f21
# CHECK-INST: fnmadd.d fs6, fs7, fs8, fs9, dyn
# CHECK-ALIAS: fnmadd.d fs6, fs7, fs8, fs9{{[[:space:]]}}
fnmadd.d f22, f23, f24, f25
# CHECK-INST: fadd.d fs10, fs11, ft8, dyn
# CHECK-ALIAS: fadd.d fs10, fs11, ft8{{[[:space:]]}}
fadd.d f26, f27, f28
# CHECK-INST: fsub.d ft9, ft10, ft11, dyn
# CHECK-ALIAS: fsub.d ft9, ft10, ft11{{[[:space:]]}}
fsub.d f29, f30, f31
# CHECK-INST: fmul.d ft0, ft1, ft2, dyn
# CHECK-ALIAS: fmul.d ft0, ft1, ft2{{[[:space:]]}}
fmul.d ft0, ft1, ft2
# CHECK-INST: fdiv.d ft3, ft4, ft5, dyn
# CHECK-ALIAS: fdiv.d ft3, ft4, ft5{{[[:space:]]}}
fdiv.d ft3, ft4, ft5
# CHECK-INST: fsqrt.d ft6, ft7, dyn
# CHECK-ALIAS: fsqrt.d ft6, ft7{{[[:space:]]}}
fsqrt.d ft6, ft7
# CHECK-INST: fcvt.s.d fs5, fs6, dyn
# CHECK-ALIAS: fcvt.s.d fs5, fs6{{[[:space:]]}}
fcvt.s.d fs5, fs6
# CHECK-INST: fcvt.w.d a4, ft11, dyn
# CHECK-ALIAS: fcvt.w.d a4, ft11{{[[:space:]]}}
fcvt.w.d a4, ft11
# CHECK-INST: fcvt.wu.d a5, ft10, dyn
# CHECK-ALIAS: fcvt.wu.d a5, ft10{{[[:space:]]}}
fcvt.wu.d a5, ft10
