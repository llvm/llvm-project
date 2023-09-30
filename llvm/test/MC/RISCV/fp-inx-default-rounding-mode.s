# RUN: llvm-mc %s -triple=riscv64 -mattr=+zdinx,+zhinx -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zdinx,+zhinx \
# RUN:     | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zdinx,+zhinx < %s \
# RUN:     | llvm-objdump -M no-aliases --mattr=+zdinx,+zhinx -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zdinx,+zhinx < %s \
# RUN:     | llvm-objdump --mattr=+zdinx,+zhinx -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ALIAS %s

# This test aims to check what the default rounding mode is for a given
# instruction if it's not specified, and ensures that it isn't printed when
# aliases are enabled but is printed otherwise. Instructions aren't listed
# exhaustively, but special attention is given to the fcvt instructions given
# that those that never round often default to frm=0b000 for historical
# reasons.
#
# These test cases are copied from fp-default-round-mode.s, but changed to use
# GPRs.

# Zfinx instructions

# CHECK-INST: fmadd.s a0, a1, a2, a3, dyn{{$}}
# CHECK-ALIAS: fmadd.s a0, a1, a2, a3{{$}}
fmadd.s a0, a1, a2, a3

# CHECK-INST: fadd.s a0, a1, a2, dyn{{$}}
# CHECK-ALIAS: fadd.s a0, a1, a2{{$}}
fadd.s a0, a1, a2

# CHECK-INST: fcvt.w.s a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.w.s a0, a0{{$}}
fcvt.w.s a0, a0

# CHECK-INST: fcvt.wu.s a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.wu.s a0, a0{{$}}
fcvt.wu.s a0, a0

# CHECK-INST: fcvt.s.w a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.w a0, a0{{$}}
fcvt.s.w a0, a0

# CHECK-INST: fcvt.s.wu a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.wu a0, a0{{$}}
fcvt.s.wu a0, a0

# CHECK-INST: fcvt.l.s a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.l.s a0, a0{{$}}
fcvt.l.s a0, a0

# CHECK-INST: fcvt.lu.s a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.lu.s a0, a0{{$}}
fcvt.lu.s a0, a0

# CHECK-INST: fcvt.s.l a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.l a0, a0{{$}}
fcvt.s.l a0, a0

# CHECK-INST: fcvt.s.lu a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.lu a0, a0{{$}}
fcvt.s.lu a0, a0

# Zdinx instructions

# CHECK-INST: fmadd.d a0, a1, a2, a3, dyn{{$}}
# CHECK-ALIAS: fmadd.d a0, a1, a2, a3{{$}}
fmadd.d a0, a1, a2, a3

# CHECK-INST: fadd.d a0, a1, a2, dyn{{$}}
# CHECK-ALIAS: fadd.d a0, a1, a2{{$}}
fadd.d a0, a1, a2

# CHECK-INST: fcvt.s.d a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.d a0, a0{{$}}
fcvt.s.d a0, a0

# FIXME: fcvt.d.s should have a default rounding mode.
# CHECK-INST: fcvt.d.s a0, a0{{$}}
# CHECK-ALIAS: fcvt.d.s a0, a0{{$}}
fcvt.d.s a0, a0

# CHECK-INST: fcvt.w.d a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.w.d a0, a0{{$}}
fcvt.w.d a0, a0

# CHECK-INST: fcvt.wu.d a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.wu.d a0, a0{{$}}
fcvt.wu.d a0, a0

# FIXME: fcvt.d.w should have a default rounding mode.
# CHECK-INST: fcvt.d.w a0, a0{{$}}
# CHECK-ALIAS: fcvt.d.w a0, a0{{$}}
fcvt.d.w a0, a0

# FIXME: fcvt.d.wu should have a default rounding mode.
# CHECK-INST: fcvt.d.wu a0, a0{{$}}
# CHECK-ALIAS: fcvt.d.wu a0, a0{{$}}
fcvt.d.wu a0, a0

# CHECK-INST: fcvt.l.d a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.l.d a0, a0{{$}}
fcvt.l.d a0, a0

# CHECK-INST: fcvt.lu.d a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.lu.d a0, a0{{$}}
fcvt.lu.d a0, a0

# CHECK-INST: fcvt.d.l a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.d.l a0, a0{{$}}
fcvt.d.l a0, a0

# CHECK-INST: fcvt.d.lu a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.d.lu a0, a0{{$}}
fcvt.d.lu a0, a0

# Zhinx instructions

# CHECK-INST: fmadd.h a0, a1, a2, a3, dyn{{$}}
# CHECK-ALIAS: fmadd.h a0, a1, a2, a3{{$}}
fmadd.h a0, a1, a2, a3

# CHECK-INST: fadd.h a0, a1, a2, dyn{{$}}
# CHECK-ALIAS: fadd.h a0, a1, a2{{$}}
fadd.h a0, a1, a2

# FIXME: fcvt.s.h should have a default rounding mode.
# CHECK-INST: fcvt.s.h a0, a0{{$}}
# CHECK-ALIAS: fcvt.s.h a0, a0{{$}}
fcvt.s.h a0, a0

# CHECK-INST: fcvt.h.s a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.s a0, a0{{$}}
fcvt.h.s a0, a0

# FIXME: fcvt.d.h should have a default rounding mode.
# CHECK-INST: fcvt.d.h a0, a0{{$}}
# CHECK-ALIAS: fcvt.d.h a0, a0{{$}}
fcvt.d.h a0, a0

# CHECK-INST: fcvt.h.d a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.d a0, a0{{$}}
fcvt.h.d a0, a0

# CHECK-INST: fcvt.w.h a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.w.h a0, a0{{$}}
fcvt.w.h a0, a0

# CHECK-INST: fcvt.wu.h a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.wu.h a0, a0{{$}}
fcvt.wu.h a0, a0

# CHECK-INST: fcvt.h.w a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.w a0, a0{{$}}
fcvt.h.w a0, a0

# CHECK-INST: fcvt.h.wu a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.wu a0, a0{{$}}
fcvt.h.wu a0, a0

# CHECK-INST: fcvt.l.h a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.l.h a0, a0{{$}}
fcvt.l.h a0, a0

# CHECK-INST: fcvt.lu.h a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.lu.h a0, a0{{$}}
fcvt.lu.h a0, a0

# CHECK-INST: fcvt.h.l a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.l a0, a0{{$}}
fcvt.h.l a0, a0

# CHECK-INST: fcvt.h.lu a0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.lu a0, a0{{$}}
fcvt.h.lu a0, a0
