# RUN: llvm-mc %s -triple=riscv32 -mattr=+f -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+f \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+f -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+f \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -d -mattr=+f -riscv-no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -d -mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d -mattr=+f -riscv-no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d -mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
##===----------------------------------------------------------------------===##

# TODO flw
# TODO fsw

# CHECK-INST: fsgnj.s ft0, ft1, ft1
# CHECK-ALIAS: fmv.s ft0, ft1
fmv.s f0, f1
# CHECK-INST: fsgnjx.s ft1, ft2, ft2
# CHECK-ALIAS: fabs.s ft1, ft2
fabs.s f1, f2
# CHECK-INST: fsgnjn.s ft2, ft3, ft3
# CHECK-ALIAS: fneg.s ft2, ft3
fneg.s f2, f3

# The following instructions actually alias instructions from the base ISA.
# However, it only makes sense to support them when the F extension is enabled.
# CHECK-INST: csrrs t0, 3, zero
# CHECK-ALIAS: frcsr t0
frcsr x5
# CHECK-INST: csrrw t1, 3, t2
# CHECK-ALIAS: fscsr t1, t2
fscsr x6, x7
# CHECK-INST: csrrw  zero, 3, t3
# CHECK-ALIAS: fscsr t3
fscsr x28

# CHECK-INST: csrrs t4, 2, zero
# CHECK-ALIAS: frrm t4
frrm x29
# CHECK-INST: csrrw  t5, 2, t4
# CHECK-ALIAS: fsrm t5, t4
fsrm x30, x29
# CHECK-INST: csrrw  zero, 2, t6
# CHECK-ALIAS: fsrm t6
fsrm x31
# CHECK-INST: csrrwi a0, 2, 31
# CHECK-ALIAS: fsrmi a0, 31
fsrmi x10, 0x1f
# CHECK-INST: csrrwi  zero, 2, 30
# CHECK-ALIAS: fsrmi 30
fsrmi 0x1e

# CHECK-INST: csrrs a1, 1, zero
# CHECK-ALIAS: frflags a1
frflags x11
# CHECK-INST: csrrw a2, 1, a1
# CHECK-ALIAS: fsflags a2, a1
fsflags x12, x11
# CHECK-INST: csrrw zero, 1, a3
# CHECK-ALIAS: fsflags a3
fsflags x13
# CHECK-INST: csrrwi a4, 1, 29
# CHECK-ALIAS: fsflagsi a4, 29
fsflagsi x14, 0x1d
# CHECK-INST: csrrwi zero, 1, 28
# CHECK-ALIAS: fsflagsi 28
fsflagsi 0x1c

##===----------------------------------------------------------------------===##
## Aliases which omit the rounding mode.
##===----------------------------------------------------------------------===##

# CHECK-INST: fmadd.s fa0, fa1, fa2, fa3, dyn
# CHECK-ALIAS: fmadd.s fa0, fa1, fa2, fa3{{[[:space:]]}}
fmadd.s f10, f11, f12, f13
# CHECK-INST: fmsub.s fa4, fa5, fa6, fa7, dyn
# CHECK-ALIAS: fmsub.s fa4, fa5, fa6, fa7{{[[:space:]]}}
fmsub.s f14, f15, f16, f17
# CHECK-INST: fnmsub.s fs2, fs3, fs4, fs5, dyn
# CHECK-ALIAS: fnmsub.s fs2, fs3, fs4, fs5{{[[:space:]]}}
fnmsub.s f18, f19, f20, f21
# CHECK-INST: fnmadd.s fs6, fs7, fs8, fs9, dyn
# CHECK-ALIAS: fnmadd.s fs6, fs7, fs8, fs9{{[[:space:]]}}
fnmadd.s f22, f23, f24, f25
# CHECK-INST: fadd.s fs10, fs11, ft8, dyn
# CHECK-ALIAS: fadd.s fs10, fs11, ft8{{[[:space:]]}}
fadd.s f26, f27, f28
# CHECK-INST: fsub.s ft9, ft10, ft11, dyn
# CHECK-ALIAS: fsub.s ft9, ft10, ft11{{[[:space:]]}}
fsub.s f29, f30, f31
# CHECK-INST: fmul.s ft0, ft1, ft2, dyn
# CHECK-ALIAS: fmul.s ft0, ft1, ft2{{[[:space:]]}}
fmul.s ft0, ft1, ft2
# CHECK-INST: fdiv.s ft3, ft4, ft5, dyn
# CHECK-ALIAS: fdiv.s ft3, ft4, ft5{{[[:space:]]}}
fdiv.s ft3, ft4, ft5
# CHECK-INST: fsqrt.s ft6, ft7, dyn
# CHECK-ALIAS: fsqrt.s ft6, ft7{{[[:space:]]}}
fsqrt.s ft6, ft7
# CHECK-INST: fcvt.w.s a0, fs5, dyn
# CHECK-ALIAS: fcvt.w.s a0, fs5{{[[:space:]]}}
fcvt.w.s a0, fs5
# CHECK-INST: fcvt.wu.s a1, fs6, dyn
# CHECK-ALIAS: fcvt.wu.s a1, fs6{{[[:space:]]}}
fcvt.wu.s a1, fs6
# CHECK-INST: fcvt.s.w ft11, a4, dyn
# CHECK-ALIAS: fcvt.s.w ft11, a4{{[[:space:]]}}
fcvt.s.w ft11, a4
# CHECK-INST: fcvt.s.wu ft0, a5, dyn
# CHECK-ALIAS: fcvt.s.wu ft0, a5{{[[:space:]]}}
fcvt.s.wu ft0, a5
