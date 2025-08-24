# RUN: not llvm-mc -triple x86_64 -mattr=+xsave,+ssse3,+avx,+tbm %s 2>&1 | \
# RUN:   FileCheck --check-prefix=ERROR --implicit-check-not=error: %s

# An extended GPR (r16-r31) can only be encoded by an EVEX
# instruction, or a legacy instruction in the primary or 0F opcode map
# (which the REX2 prefix can extend). Instructions outside that set
# reject an r16-r31 in any operand.

# XSAVE/XRSTOR are in the 0F map but are excepted from REX2 promotion,
# so the address base cannot be an EGPR.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
xsave (%r16)
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
xrstor (%r16)

# PABSB is in the 0F38 map, so neither the base nor the index of its
# address may be an EGPR.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pabsb (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pabsb (%rax,%r16), %xmm0

# TBM is XOP-encoded.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
bextr $1, (%r16), %eax
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
blcfill %r16d, %eax

# The VEX form of cvtsd2si cannot write an EGPR. The EVEX form can,
# but only with AVX512, so force VEX to select the restricted
# encoding.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
{vex} vcvtsd2si %xmm0, %r16d
