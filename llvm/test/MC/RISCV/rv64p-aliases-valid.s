# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-p -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-p \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-p < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r -M no-aliases --mattr=+experimental-p - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-p < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r --mattr=+experimental-p - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# Tests for aliases that are available on both RV32 and RV64

# CHECK-S-OBJ-NOALIAS: padd.bs a0, zero, a1
# CHECK-S-OBJ: pmv.bs a0, a1
pmv.bs a0, a1

# CHECK-S-OBJ-NOALIAS: padd.hs a2, zero, a3
# CHECK-S-OBJ: pmv.hs a2, a3
pmv.hs a2, a3

# CHECK-S-OBJ-NOALIAS: psub.b a0, zero, a1
# CHECK-S-OBJ: pneg.b a0, a1
pneg.b a0, a1

# CHECK-S-OBJ-NOALIAS: psub.h a2, zero, a3
# CHECK-S-OBJ: pneg.h a2, a3
pneg.h a2, a3

# CHECK-S-OBJ-NOALIAS: pabd.b a0, a1, zero
# CHECK-S-OBJ: pabs.b a0, a1
pabs.b a0, a1

# CHECK-S-OBJ-NOALIAS: pabd.h a2, a3, zero
# CHECK-S-OBJ: pabs.h a2, a3
pabs.h a2, a3

# CHECK-S-OBJ-NOALIAS: ppaire.b a0, a1, zero
# CHECK-S-OBJ: pzext.h.b a0, a1
pzext.h.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmseq.b a0, a1, zero
# CHECK-S-OBJ: pmseqz.b a0, a1
pmseqz.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmseq.h a2, a3, zero
# CHECK-S-OBJ: pmseqz.h a2, a3
pmseqz.h a2, a3

# CHECK-S-OBJ-NOALIAS: pmsltu.b a0, zero, a1
# CHECK-S-OBJ: pmsnez.b a0, a1
pmsnez.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmsltu.h a2, zero, a3
# CHECK-S-OBJ: pmsnez.h a2, a3
pmsnez.h a2, a3

# CHECK-S-OBJ-NOALIAS: pmslt.b a0, a1, zero
# CHECK-S-OBJ: pmsltz.b a0, a1
pmsltz.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmslt.h a2, a3, zero
# CHECK-S-OBJ: pmsltz.h a2, a3
pmsltz.h a2, a3

# CHECK-S-OBJ-NOALIAS: pmslt.b a0, zero, a1
# CHECK-S-OBJ: pmsgtz.b a0, a1
pmsgtz.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmslt.h a2, zero, a3
# CHECK-S-OBJ: pmsgtz.h a2, a3
pmsgtz.h a2, a3

# CHECK-S-OBJ-NOALIAS: pmslt.b a0, a2, a1
# CHECK-S-OBJ: pmslt.b a0, a2, a1
pmsgt.b a0, a1, a2

# CHECK-S-OBJ-NOALIAS: pmslt.h a3, a5, a4
# CHECK-S-OBJ: pmslt.h a3, a5, a4
pmsgt.h a3, a4, a5

# CHECK-S-OBJ-NOALIAS: pmsltu.b a0, a2, a1
# CHECK-S-OBJ: pmsltu.b a0, a2, a1
pmsgtu.b a0, a1, a2

# CHECK-S-OBJ-NOALIAS: pmsltu.h a3, a5, a4
# CHECK-S-OBJ: pmsltu.h a3, a5, a4
pmsgtu.h a3, a4, a5

# Tests for RV64-only aliases (word operations)

# CHECK-S-OBJ-NOALIAS: padd.ws a0, zero, a1
# CHECK-S-OBJ: pmv.ws a0, a1
pmv.ws a0, a1

# CHECK-S-OBJ-NOALIAS: psub.w a2, zero, a3
# CHECK-S-OBJ: pneg.w a2, a3
pneg.w a2, a3

# CHECK-S-OBJ-NOALIAS: ppaire.h a0, a1, zero
# CHECK-S-OBJ: pzext.w.h a0, a1
pzext.w.h a0, a1

# CHECK-S-OBJ-NOALIAS: pmseq.w a0, a1, zero
# CHECK-S-OBJ: pmseqz.w a0, a1
pmseqz.w a0, a1

# CHECK-S-OBJ-NOALIAS: pmsltu.w a0, zero, a1
# CHECK-S-OBJ: pmsnez.w a0, a1
pmsnez.w a0, a1

# CHECK-S-OBJ-NOALIAS: pmslt.w a0, a1, zero
# CHECK-S-OBJ: pmsltz.w a0, a1
pmsltz.w a0, a1

# CHECK-S-OBJ-NOALIAS: pmslt.w a0, zero, a1
# CHECK-S-OBJ: pmsgtz.w a0, a1
pmsgtz.w a0, a1

# CHECK-S-OBJ-NOALIAS: pmslt.w a0, a2, a1
# CHECK-S-OBJ: pmslt.w a0, a2, a1
pmsgt.w a0, a1, a2

# CHECK-S-OBJ-NOALIAS: pmsltu.w a3, a5, a4
# CHECK-S-OBJ: pmsltu.w a3, a5, a4
pmsgtu.w a3, a4, a5

# CHECK-S-OBJ-NOALIAS: zip8p a4, a5, zero
# CHECK-S-OBJ: pwcvtu.b a4, a5
pwcvtu.b a4, a5

# CHECK-S-OBJ-NOALIAS: zip16p a6, a7, zero
# CHECK-S-OBJ: pwcvtu.h a6, a7
pwcvtu.h a6, a7

# CHECK-S-OBJ-NOALIAS: zip8p s0, zero, s1
# CHECK-S-OBJ: pwcvth.b s0, s1
pwcvth.b s0, s1

# CHECK-S-OBJ-NOALIAS: zip16p s2, zero, s3
# CHECK-S-OBJ: pwcvth.h s2, s3
pwcvth.h s2, s3

# CHECK-S-OBJ-NOALIAS: unzip8p s3, s4, zero
# CHECK-S-OBJ: pncvt.b s3, s4
pncvt.b s3, s4

# CHECK-S-OBJ-NOALIAS: unzip16p s5, s6, zero
# CHECK-S-OBJ: pncvt.h s5, s6
pncvt.h s5, s6

# CHECK-S-OBJ-NOALIAS: unzip8hp s7, s8, zero
# CHECK-S-OBJ: pncvth.b s7, s8
pncvth.b s7, s8

# CHECK-S-OBJ-NOALIAS: unzip16hp s9, s10, zero
# CHECK-S-OBJ: pncvth.h s9, s10
pncvth.h s9, s10

# CHECK-S-OBJ-NOALIAS: pli.b s10, 17
# CHECK-S-OBJ: pli.b s10, 17
li s10, 0x1111111111111111

# CHECK-S-OBJ-NOALIAS: pli.h s11, 1
# CHECK-S-OBJ: pli.h s11, 1
li s11, 0x0001000100010001

# CHECK-S-OBJ-NOALIAS: pli.w t3, 1
# CHECK-S-OBJ: pli.w t3, 1
li t3, 0x0000000100000001
