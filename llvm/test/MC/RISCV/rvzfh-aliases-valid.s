# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfh -M no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfh \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfh -M no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfh \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zfh < %s \
# RUN:     | llvm-objdump -d --mattr=+zfh --no-print-imm-hex -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zfh < %s \
# RUN:     | llvm-objdump -d --mattr=+zfh --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zfh < %s \
# RUN:     | llvm-objdump -d --mattr=+zfh --no-print-imm-hex -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zfh < %s \
# RUN:     | llvm-objdump -d --mattr=+zfh --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
##===----------------------------------------------------------------------===##

# CHECK-INST: fsgnj.h ft0, ft1, ft1
# CHECK-ALIAS: fmv.h ft0, ft1
fmv.h f0, f1
# CHECK-INST: fsgnjx.h ft1, ft2, ft2
# CHECK-ALIAS: fabs.h ft1, ft2
fabs.h f1, f2
# CHECK-INST: fsgnjn.h ft2, ft3, ft3
# CHECK-ALIAS: fneg.h ft2, ft3
fneg.h f2, f3

# CHECK-INST: flt.h tp, ft6, ft5
# CHECK-ALIAS: flt.h tp, ft6, ft5
fgt.h x4, f5, f6
# CHECK-INST: fle.h t2, fs1, fs0
# CHECK-ALIAS: fle.h t2, fs1, fs0
fge.h x7, f8, f9

# CHECK-INST: fmv.x.h a2, fs7
# CHECK-ALIAS: fmv.x.h a2, fs7
fmv.x.h a2, fs7
# CHECK-INST: fmv.h.x ft1, a6
# CHECK-ALIAS: fmv.h.x ft1, a6
fmv.h.x ft1, a6

# CHECK-INST: flh ft0, 0(a0)
# CHECK-ALIAS: flh ft0, 0(a0)
flh f0, (x10)
# CHECK-INST: fsh ft0, 0(a0)
# CHECK-ALIAS: fsh ft0, 0(a0)
fsh f0, (x10)

##===----------------------------------------------------------------------===##
## Aliases which omit the rounding mode.
##===----------------------------------------------------------------------===##

# CHECK-INST: fmadd.h fa0, fa1, fa2, fa3, dyn
# CHECK-ALIAS: fmadd.h fa0, fa1, fa2, fa3{{[[:space:]]}}
fmadd.h f10, f11, f12, f13
# CHECK-INST: fmsub.h fa4, fa5, fa6, fa7, dyn
# CHECK-ALIAS: fmsub.h fa4, fa5, fa6, fa7{{[[:space:]]}}
fmsub.h f14, f15, f16, f17
# CHECK-INST: fnmsub.h fs2, fs3, fs4, fs5, dyn
# CHECK-ALIAS: fnmsub.h fs2, fs3, fs4, fs5{{[[:space:]]}}
fnmsub.h f18, f19, f20, f21
# CHECK-INST: fnmadd.h fs6, fs7, fs8, fs9, dyn
# CHECK-ALIAS: fnmadd.h fs6, fs7, fs8, fs9{{[[:space:]]}}
fnmadd.h f22, f23, f24, f25
# CHECK-INST: fadd.h fs10, fs11, ft8, dyn
# CHECK-ALIAS: fadd.h fs10, fs11, ft8{{[[:space:]]}}
fadd.h f26, f27, f28
# CHECK-INST: fsub.h ft9, ft10, ft11, dyn
# CHECK-ALIAS: fsub.h ft9, ft10, ft11{{[[:space:]]}}
fsub.h f29, f30, f31
# CHECK-INST: fmul.h ft0, ft1, ft2, dyn
# CHECK-ALIAS: fmul.h ft0, ft1, ft2{{[[:space:]]}}
fmul.h ft0, ft1, ft2
# CHECK-INST: fdiv.h ft3, ft4, ft5, dyn
# CHECK-ALIAS: fdiv.h ft3, ft4, ft5{{[[:space:]]}}
fdiv.h ft3, ft4, ft5
# CHECK-INST: fsqrt.h ft6, ft7, dyn
# CHECK-ALIAS: fsqrt.h ft6, ft7{{[[:space:]]}}
fsqrt.h ft6, ft7
# CHECK-INST: fcvt.w.h a0, fs5, dyn
# CHECK-ALIAS: fcvt.w.h a0, fs5{{[[:space:]]}}
fcvt.w.h a0, fs5
# CHECK-INST: fcvt.wu.h a1, fs6, dyn
# CHECK-ALIAS: fcvt.wu.h a1, fs6{{[[:space:]]}}
fcvt.wu.h a1, fs6
# CHECK-INST: fcvt.h.w ft11, a4, dyn
# CHECK-ALIAS: fcvt.h.w ft11, a4{{[[:space:]]}}
fcvt.h.w ft11, a4
# CHECK-INST: fcvt.h.wu ft0, a5, dyn
# CHECK-ALIAS: fcvt.h.wu ft0, a5{{[[:space:]]}}
fcvt.h.wu ft0, a5
