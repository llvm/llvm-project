# RUN: llvm-mc %s -triple=riscv32 -mattr=+zhinx -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zhinx \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zhinx -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zhinx \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zhinx %s \
# RUN:     | llvm-objdump -d --mattr=+zhinx -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zhinx %s \
# RUN:     | llvm-objdump -d --mattr=+zhinx - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zhinx %s \
# RUN:     | llvm-objdump -d --mattr=+zhinx -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zhinx %s \
# RUN:     | llvm-objdump -d --mattr=+zhinx - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
##===----------------------------------------------------------------------===##

# CHECK-INST: fsgnjx.h s1, s2, s2
# CHECK-ALIAS: fabs.h s1, s2
fabs.h s1, s2
# CHECK-INST: fsgnjn.h s2, s3, s3
# CHECK-ALIAS: fneg.h s2, s3
fneg.h s2, s3

# CHECK-INST: flt.h tp, s6, s5
# CHECK-ALIAS: flt.h tp, s6, s5
fgt.h x4, s5, s6
# CHECK-INST: fle.h t2, s1, s0
# CHECK-ALIAS: fle.h t2, s1, s0
fge.h x7, x8, x9

##===----------------------------------------------------------------------===##
## Aliases which omit the rounding mode.
##===----------------------------------------------------------------------===##

# CHECK-INST: fmadd.h a0, a1, a2, a3, dyn
# CHECK-ALIAS: fmadd.h a0, a1, a2, a3
fmadd.h x10, x11, x12, x13
# CHECK-INST: fmsub.h a4, a5, a6, a7, dyn
# CHECK-ALIAS: fmsub.h a4, a5, a6, a7
fmsub.h x14, x15, x16, x17
# CHECK-INST: fnmsub.h s2, s3, s4, s5, dyn
# CHECK-ALIAS: fnmsub.h s2, s3, s4, s5
fnmsub.h x18, x19, x20, x21
# CHECK-INST: fnmadd.h s6, s7, s8, s9, dyn
# CHECK-ALIAS: fnmadd.h s6, s7, s8, s9
fnmadd.h x22, x23, x24, x25
# CHECK-INST: fadd.h s10, s11, t3, dyn
# CHECK-ALIAS: fadd.h s10, s11, t3
fadd.h x26, x27, x28
# CHECK-INST: fsub.h t4, t5, t6, dyn
# CHECK-ALIAS: fsub.h t4, t5, t6
fsub.h x29, x30, x31
# CHECK-INST: fmul.h s0, s1, s2, dyn
# CHECK-ALIAS: fmul.h s0, s1, s2
fmul.h s0, s1, s2
# CHECK-INST: fdiv.h s3, s4, s5, dyn
# CHECK-ALIAS: fdiv.h s3, s4, s5
fdiv.h s3, s4, s5
# CHECK-INST: fsqrt.h s6, s7, dyn
# CHECK-ALIAS: fsqrt.h s6, s7
fsqrt.h s6, s7
# CHECK-INST: fcvt.w.h a0, s5, dyn
# CHECK-ALIAS: fcvt.w.h a0, s5
fcvt.w.h a0, s5
# CHECK-INST: fcvt.wu.h a1, s6, dyn
# CHECK-ALIAS: fcvt.wu.h a1, s6
fcvt.wu.h a1, s6
# CHECK-INST: fcvt.h.w t6, a4, dyn
# CHECK-ALIAS: fcvt.h.w t6, a4
fcvt.h.w t6, a4
# CHECK-INST: fcvt.h.wu s0, a5, dyn
# CHECK-ALIAS: fcvt.h.wu s0, a5
fcvt.h.wu s0, a5
