# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfinx -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfinx \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfinx -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfinx \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zfinx %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d --mattr=+zfinx -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zfinx %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d --mattr=+zfinx - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zfinx %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d --mattr=+zfinx -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zfinx %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d --mattr=+zfinx - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
##===----------------------------------------------------------------------===##

# CHECK-INST: fsgnj.s s1, s2, s2
# CHECK-ALIAS: fmv.s s1, s2
fmv.s s1, s2
# CHECK-INST: fsgnjx.s s1, s2, s2
# CHECK-ALIAS: fabs.s s1, s2
fabs.s s1, s2
# CHECK-INST: fsgnjn.s s2, s3, s3
# CHECK-ALIAS: fneg.s s2, s3
fneg.s s2, s3

# CHECK-INST: flt.s tp, s6, s5
# CHECK-ALIAS: flt.s tp, s6, s5
fgt.s x4, s5, s6
# CHECK-INST: fle.s t2, s1, s0
# CHECK-ALIAS: fle.s t2, s1, s0
fge.s x7, x8, x9

# The following instructions actually alias instructions from the base ISA.
# However, it only makes sense to support them when the F or Zfinx extension is
# enabled.
# CHECK-INST: csrrs t0, fcsr, zero
# CHECK-ALIAS: frcsr t0
frcsr x5
# CHECK-INST: csrrw t1, fcsr, t2
# CHECK-ALIAS: fscsr t1, t2
fscsr x6, x7
# CHECK-INST: csrrw  zero, fcsr, t3
# CHECK-ALIAS: fscsr t3
fscsr x28

# These are obsolete aliases of frcsr/fscsr. They are accepted by the assembler
# but the disassembler should always print them as the equivalent, new aliases.
# CHECK-INST: csrrs t4, fcsr, zero
# CHECK-ALIAS: frcsr t4
frsr x29
# CHECK-INST: csrrw t5, fcsr, t6
# CHECK-ALIAS: fscsr t5, t6
fssr x30, x31
# CHECK-INST: csrrw zero, fcsr, s0
# CHECK-ALIAS: fscsr s0
fssr x8

# CHECK-INST: csrrs t4, frm, zero
# CHECK-ALIAS: frrm t4
frrm x29
# CHECK-INST: csrrw  t5, frm, t4
# CHECK-ALIAS: fsrm t5, t4
fsrm x30, x29
# CHECK-INST: csrrw  zero, frm, t6
# CHECK-ALIAS: fsrm t6
fsrm x31
# CHECK-INST: csrrwi a0, frm, 31
# CHECK-ALIAS: fsrmi a0, 31
fsrmi x10, 0x1f
# CHECK-INST: csrrwi  zero, frm, 30
# CHECK-ALIAS: fsrmi 30
fsrmi 0x1e

# CHECK-INST: csrrs a1, fflags, zero
# CHECK-ALIAS: frflags a1
frflags x11
# CHECK-INST: csrrw a2, fflags, a1
# CHECK-ALIAS: fsflags a2, a1
fsflags x12, x11
# CHECK-INST: csrrw zero, fflags, a3
# CHECK-ALIAS: fsflags a3
fsflags x13
# CHECK-INST: csrrwi a4, fflags, 29
# CHECK-ALIAS: fsflagsi a4, 29
fsflagsi x14, 0x1d
# CHECK-INST: csrrwi zero, fflags, 28
# CHECK-ALIAS: fsflagsi 28
fsflagsi 0x1c

##===----------------------------------------------------------------------===##
## Aliases which omit the rounding mode.
##===----------------------------------------------------------------------===##

# CHECK-INST: fmadd.s a0, a1, a2, a3, dyn
# CHECK-ALIAS: fmadd.s a0, a1, a2, a3
fmadd.s x10, x11, x12, x13
# CHECK-INST: fmsub.s a4, a5, a6, a7, dyn
# CHECK-ALIAS: fmsub.s a4, a5, a6, a7
fmsub.s x14, x15, x16, x17
# CHECK-INST: fnmsub.s s2, s3, s4, s5, dyn
# CHECK-ALIAS: fnmsub.s s2, s3, s4, s5
fnmsub.s x18, x19, x20, x21
# CHECK-INST: fnmadd.s s6, s7, s8, s9, dyn
# CHECK-ALIAS: fnmadd.s s6, s7, s8, s9
fnmadd.s x22, x23, x24, x25
# CHECK-INST: fadd.s s10, s11, t3, dyn
# CHECK-ALIAS: fadd.s s10, s11, t3
fadd.s x26, x27, x28
# CHECK-INST: fsub.s t4, t5, t6, dyn
# CHECK-ALIAS: fsub.s t4, t5, t6
fsub.s x29, x30, x31
# CHECK-INST: fmul.s s0, s1, s2, dyn
# CHECK-ALIAS: fmul.s s0, s1, s2
fmul.s s0, s1, s2
# CHECK-INST: fdiv.s s3, s4, s5, dyn
# CHECK-ALIAS: fdiv.s s3, s4, s5
fdiv.s s3, s4, s5
# CHECK-INST: sqrt.s s6, s7, dyn
# CHECK-ALIAS: sqrt.s s6, s7
fsqrt.s s6, s7
# CHECK-INST: fcvt.w.s a0, s5, dyn
# CHECK-ALIAS: fcvt.w.s a0, s5
fcvt.w.s a0, s5
# CHECK-INST: fcvt.wu.s a1, s6, dyn
# CHECK-ALIAS: fcvt.wu.s a1, s6
fcvt.wu.s a1, s6
# CHECK-INST: fcvt.s.w t6, a4, dyn
# CHECK-ALIAS: fcvt.s.w t6, a4
fcvt.s.w t6, a4
# CHECK-INST: fcvt.s.wu s0, a5, dyn
# CHECK-ALIAS: fcvt.s.wu s0, a5
fcvt.s.wu s0, a5
