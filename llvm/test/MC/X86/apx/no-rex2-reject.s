# RUN: not llvm-mc -triple x86_64 -mattr=+xsave,+ssse3,+sse4.2,+avx,+avx2,+tbm,+aes,+vaes,+sha,+pclmul,+gfni,+kl,+widekl %s 2>&1 | \
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

# The legacy-vector 0F38/0F3A maps have no EVEX form, so their address
# operands cannot use an EGPR. A representative from each affected family:

# AES (0F38).
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
aesenc (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
aeskeygenassist $1, (%r16), %xmm0

# Key Locker (0F38).
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
aesenc128kl (%r16), %xmm0

# SHA (0F38/0F3A).
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
sha256rnds2 (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
sha1rnds4 $1, (%r16), %xmm0

# PCLMULQDQ (0F3A).
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pclmulqdq $1, (%r16), %xmm0

# GFNI (0F38/0F3A).
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
gf2p8mulb (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
gf2p8affineqb $1, (%r16), %xmm0

# SSE4 integer/blend/pack/compare (0F38/0F3A).
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pmulld (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
blendps $1, (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
packusdw (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pcmpgtq (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
mpsadbw $1, (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
roundps $1, (%r16), %xmm0
# The scalar intrinsic forms take an ssmem/sdmem source.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
roundss $1, (%r16), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
roundsd $1, (%r16), %xmm0

# The index register is equally restricted.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pshufb (%rax,%r16), %xmm0

# PMOVSX/ZX (0F38).
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pmovsxbw (%r16), %xmm0

# PEXTR/PINSR and EXTRACTPS have a direct GPR operand that likewise
# cannot be an EGPR in the legacy encoding.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pextrd $1, %xmm0, %r16d
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pinsrd $1, %r16d, %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
extractps $1, %xmm0, %r16d
# The 64-bit forms constrain the GPR the same way.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pextrq $1, %xmm0, %r16
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
pinsrq $1, %r16, %xmm0

# VEX gather and maskmov have no in-place EVEX promotion, so the address
# base cannot be an EGPR.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
vgatherdps %xmm1, (%r16,%xmm2), %xmm0
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
vpmaskmovd %xmm1, %xmm2, (%r16)
# The 256-bit maskmov store uses a ymm-sized memory operand.
# ERROR: [[#@LINE+1]]:1: error: unsupported instruction
vmaskmovps %ymm1, %ymm2, (%r16)
