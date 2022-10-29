# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv65 -filetype=obj %s | llvm-objdump --no-print-imm-hex --mcpu=hexagonv65 -d - | FileCheck --implicit-check-not='{' %s

# This case requires compounding only some of the instructions which are
# possible compounds.  Compounding all possible opcodes is ideal for code size
# but does not always result in a packet with a valid shuffle, whereas the
# non-compounded instructions may be a valid shuffle.

foo:
{ r0=c0
  p0=cmp.eq(r0,#0); if (p0.new) jump:nt foo
  jump foo
  r1=r0 }

# CHECK-LABEL:  <foo>:
# CHECK-NEXT:  { r0 = sa0
# CHECK-NEXT:    p0 = cmp.eq(r0,#0); if (p0.new) jump:nt 0x0
# CHECK-NEXT:    jump 0x0
# CHECK-NEXT:    r1 = r0 }

