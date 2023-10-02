# RUN: llvm-mc %s -triple=riscv64 -mattr=+d,+zfh,+experimental-zfbfmin -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+d,+zfh,+experimental-zfbfmin \
# RUN:     | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+d,+zfh,+experimental-zfbfmin < %s \
# RUN:     | llvm-objdump -M no-aliases --mattr=+d,+zfh,+experimental-zfbfmin -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+d,+zfh,+experimental-zfbfmin < %s \
# RUN:     | llvm-objdump --mattr=+d,+zfh,+experimental-zfbfmin -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ALIAS %s

# This test aims to check what the default rounding mode is for a given
# instruction if it's not specified, and ensures that it isn't printed when
# aliases are enabled but is printed otherwise. Instructions aren't listed
# exhaustively, but special attention is given to the fcvt instructions given
# that those that never round often default to frm=0b000 for historical
# reasons.

# F instructions

# CHECK-INST: fmadd.s fa0, fa1, fa2, fa3, dyn{{$}}
# CHECK-ALIAS: fmadd.s fa0, fa1, fa2, fa3{{$}}
fmadd.s fa0, fa1, fa2, fa3

# CHECK-INST: fadd.s fa0, fa1, fa2, dyn{{$}}
# CHECK-ALIAS: fadd.s fa0, fa1, fa2{{$}}
fadd.s fa0, fa1, fa2

# CHECK-INST: fcvt.w.s a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.w.s a0, fa0{{$}}
fcvt.w.s a0, fa0

# CHECK-INST: fcvt.wu.s a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.wu.s a0, fa0{{$}}
fcvt.wu.s a0, fa0

# CHECK-INST: fcvt.s.w fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.w fa0, a0{{$}}
fcvt.s.w fa0, a0

# CHECK-INST: fcvt.s.wu fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.wu fa0, a0{{$}}
fcvt.s.wu fa0, a0

# CHECK-INST: fcvt.l.s a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.l.s a0, fa0{{$}}
fcvt.l.s a0, fa0

# CHECK-INST: fcvt.lu.s a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.lu.s a0, fa0{{$}}
fcvt.lu.s a0, fa0

# CHECK-INST: fcvt.s.l fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.l fa0, a0{{$}}
fcvt.s.l fa0, a0

# CHECK-INST: fcvt.s.lu fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.lu fa0, a0{{$}}
fcvt.s.lu fa0, a0

# D instructions

# CHECK-INST: fmadd.d fa0, fa1, fa2, fa3, dyn{{$}}
# CHECK-ALIAS: fmadd.d fa0, fa1, fa2, fa3{{$}}
fmadd.d fa0, fa1, fa2, fa3

# CHECK-INST: fadd.d fa0, fa1, fa2, dyn{{$}}
# CHECK-ALIAS: fadd.d fa0, fa1, fa2{{$}}
fadd.d fa0, fa1, fa2

# CHECK-INST: fcvt.s.d fa0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.d fa0, fa0{{$}}
fcvt.s.d fa0, fa0

# For historical reasons defaults to frm==0b000 (rne) but doesn't print this
# default rounding mode.
# CHECK-INST: fcvt.d.s fa0, fa0{{$}}
# CHECK-ALIAS: fcvt.d.s fa0, fa0{{$}}
fcvt.d.s fa0, fa0
# CHECK-INST: fcvt.d.s fa0, fa0{{$}}
# CHECK-ALIAS: fcvt.d.s fa0, fa0{{$}}
fcvt.d.s fa0, fa0, rne

# CHECK-INST: fcvt.w.d a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.w.d a0, fa0{{$}}
fcvt.w.d a0, fa0

# CHECK-INST: fcvt.wu.d a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.wu.d a0, fa0{{$}}
fcvt.wu.d a0, fa0

# For historical reasons defaults to frm==0b000 (rne) but doesn't print this
# default rounding mode.
# CHECK-INST: fcvt.d.w fa0, a0{{$}}
# CHECK-ALIAS: fcvt.d.w fa0, a0{{$}}
fcvt.d.w fa0, a0
# CHECK-INST: fcvt.d.w fa0, a0{{$}}
# CHECK-ALIAS: fcvt.d.w fa0, a0{{$}}
fcvt.d.w fa0, a0, rne

# For historical reasons defaults to frm==0b000 (rne) but doesn't print this
# default rounding mode.
# CHECK-INST: fcvt.d.wu fa0, a0{{$}}
# CHECK-ALIAS: fcvt.d.wu fa0, a0{{$}}
fcvt.d.wu fa0, a0
# CHECK-INST: fcvt.d.wu fa0, a0{{$}}
# CHECK-ALIAS: fcvt.d.wu fa0, a0{{$}}
fcvt.d.wu fa0, a0, rne

# CHECK-INST: fcvt.l.d a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.l.d a0, fa0{{$}}
fcvt.l.d a0, fa0

# CHECK-INST: fcvt.lu.d a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.lu.d a0, fa0{{$}}
fcvt.lu.d a0, fa0

# CHECK-INST: fcvt.d.l fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.d.l fa0, a0{{$}}
fcvt.d.l fa0, a0

# CHECK-INST: fcvt.d.lu fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.d.lu fa0, a0{{$}}
fcvt.d.lu fa0, a0

# Zfh instructions

# CHECK-INST: fmadd.h fa0, fa1, fa2, fa3, dyn{{$}}
# CHECK-ALIAS: fmadd.h fa0, fa1, fa2, fa3{{$}}
fmadd.h fa0, fa1, fa2, fa3

# CHECK-INST: fadd.h fa0, fa1, fa2, dyn{{$}}
# CHECK-ALIAS: fadd.h fa0, fa1, fa2{{$}}
fadd.h fa0, fa1, fa2

# For historical reasons defaults to frm==0b000 (rne) but doesn't print this
# default rounding mode.
# CHECK-INST: fcvt.s.h fa0, fa0{{$}}
# CHECK-ALIAS: fcvt.s.h fa0, fa0{{$}}
fcvt.s.h fa0, fa0
# CHECK-INST: fcvt.s.h fa0, fa0{{$}}
# CHECK-ALIAS: fcvt.s.h fa0, fa0{{$}}
fcvt.s.h fa0, fa0, rne

# CHECK-INST: fcvt.h.s fa0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.s fa0, fa0{{$}}
fcvt.h.s fa0, fa0

# For historical reasons defaults to frm==0b000 (rne) but doesn't print this
# default rounding mode.
# CHECK-INST: fcvt.d.h fa0, fa0{{$}}
# CHECK-ALIAS: fcvt.d.h fa0, fa0{{$}}
fcvt.d.h fa0, fa0
# CHECK-INST: fcvt.d.h fa0, fa0{{$}}
# CHECK-ALIAS: fcvt.d.h fa0, fa0{{$}}
fcvt.d.h fa0, fa0, rne

# CHECK-INST: fcvt.h.d fa0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.d fa0, fa0{{$}}
fcvt.h.d fa0, fa0

# CHECK-INST: fcvt.w.h a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.w.h a0, fa0{{$}}
fcvt.w.h a0, fa0

# CHECK-INST: fcvt.wu.h a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.wu.h a0, fa0{{$}}
fcvt.wu.h a0, fa0

# CHECK-INST: fcvt.h.w fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.w fa0, a0{{$}}
fcvt.h.w fa0, a0

# CHECK-INST: fcvt.h.wu fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.wu fa0, a0{{$}}
fcvt.h.wu fa0, a0

# CHECK-INST: fcvt.l.h a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.l.h a0, fa0{{$}}
fcvt.l.h a0, fa0

# CHECK-INST: fcvt.lu.h a0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.lu.h a0, fa0{{$}}
fcvt.lu.h a0, fa0

# CHECK-INST: fcvt.h.l fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.l fa0, a0{{$}}
fcvt.h.l fa0, a0

# CHECK-INST: fcvt.h.lu fa0, a0, dyn{{$}}
# CHECK-ALIAS: fcvt.h.lu fa0, a0{{$}}
fcvt.h.lu fa0, a0

# Zfbfmin instructions

# CHECK-INST: fcvt.s.bf16 fa0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.s.bf16 fa0, fa0{{$}}
fcvt.s.bf16 fa0, fa0

# CHECK-INST: fcvt.bf16.s fa0, fa0, dyn{{$}}
# CHECK-ALIAS: fcvt.bf16.s fa0, fa0{{$}}
fcvt.bf16.s fa0, fa0
