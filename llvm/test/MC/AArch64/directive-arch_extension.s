// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s | FileCheck %s

.arch_extension crc
crc32cx w0, w1, x3
// CHECK: crc32cx w0, w1, x3

.arch_extension sm4
sm4e v2.4s, v15.4s
// CHECK: sm4e v2.4s, v15.4s

.arch_extension sha3
sha512h q0, q1, v2.2d
// CHECK: sha512h q0, q1, v2.2d

.arch_extension sha2
sha1h s0, s1
// CHECK: sha1h s0, s1

.arch_extension aes
aese v0.16b, v1.16b
// CHECK: aese v0.16b, v1.16b

.arch_extension fp
fminnm d0, d0, d1
// CHECK: fminnm d0, d0, d1

.arch_extension simd
addp v0.4s, v0.4s, v0.4s
// CHECK: addp v0.4s, v0.4s, v0.4s

.arch_extension ras
esb
// CHECK: esb

.arch_extension lse
casa w5, w7, [x20]
// CHECK: casa w5, w7, [x20]

.arch_extension lse128
swpp x0, x2, [x3]
// CHECK: swpp x0, x2, [x3]

.arch_extension predres
cfp rctx, x0
// CHECK: cfp rctx, x0

.arch_extension predres2
cosp rctx, x0
// CHECK: cosp rctx, x0

.arch_extension ccdp
dc cvadp, x7
// CHECK: dc cvadp, x7

.arch_extension mte
irg x0, x1
// CHECK: irg x0, x1

.arch_extension memtag
irg x0, x1
// CHECK: irg x0, x1

.arch_extension tlb-rmi
tlbi vmalle1os
// CHECK: tlbi vmalle1os

.arch_extension pan
mrs x0, pan
// CHECK: mrs x0, PAN

.arch_extension pan-rwv
at s1e1wp, x2
// CHECK: at s1e1wp, x2

.arch_extension ccpp
dc cvap, x7
// CHECK: dc cvap, x7

.arch_extension rcpc
ldapr x0, [x1]
// CHECK: ldapr x0, [x1]

.arch_extension rcpc3
stilp   w24, w0, [x16, #-8]!
// CHECK: stilp   w24, w0, [x16, #-8]!

.arch_extension ls64
ld64b x0, [x13]
// CHECK: ld64b x0, [x13]

.arch_extension pauth
paciasp
// CHECK: paciasp

.arch_extension flagm
cfinv
// CHECK: cfinv

.arch_extension hbc
lbl:
    bc.eq lbl
// CHECK: bc.eq lbl

.arch_extension mops
cpyfp [x0]!, [x1]!, x2!
// CHECK: cpyfp [x0]!, [x1]!, x2!

.arch_extension the
rcwswp x0, x1, [x2]
// CHECK: rcwswp x0, x1, [x2]

// This needs to come after `.arch_extension the` as it uses an instruction that
// requires both the and d128
.arch_extension d128
sysp #0, c2, c0, #0, x0, x1
rcwcasp   x0, x1, x6, x7, [x4]
// CHECK: sysp #0, c2, c0, #0, x0, x1
// CHECK: rcwcasp   x0, x1, x6, x7, [x4]

.arch_extension rasv2
mrs x0, ERXGSR_EL1
// CHECK: mrs x0, ERXGSR_EL1

.arch_extension ite
trcit x0
// CHECK: trcit x0

.arch_extension cssc
umax x0, x1, x2
// CHECK: umax x0, x1, x2

.arch_extension gcs
gcspushm x0
// CHECK: gcspushm x0

.arch_extension bf16
bfdot v0.2s, v0.4h, v0.4h
// CHECK: bfdot v0.2s, v0.4h, v0.4h

.arch_extension compnum
fcmla v1.2d, v2.2d, v3.2d, #0
// CHECK: fcmla v1.2d, v2.2d, v3.2d, #0

.arch_extension dotprod
udot v0.4s, v0.16b, v0.16b
// CHECK: udot v0.4s, v0.16b, v0.16b

.arch_extension f32mm
fmmla z0.s, z1.s, z2.s
// CHECK: fmmla z0.s, z1.s, z2.s

.arch_extension f64mm
fmmla z0.d, z1.d, z2.d
// CHECK: fmmla z0.d, z1.d, z2.d

.arch_extension fp16
fadd v0.8h, v0.8h, v0.8h
// CHECK: fadd v0.8h, v0.8h, v0.8h

.arch_extension fp16fml
fmlal v0.2s, v1.2h, v2.2h
// CHECK: fmlal v0.2s, v1.2h, v2.2h

.arch_extension i8mm
usdot v0.4s, v0.16b, v0.16b
// CHECK: usdot v0.4s, v0.16b, v0.16b

.arch_extension lor
stllr x0, [x0]
// CHECK: stllr x0, [x0]

.arch_extension profile
msr PMBLIMITR_EL1, x0
// CHECK: msr PMBLIMITR_EL1, x0

.arch_extension rdm
.arch_extension rdma
sqrdmlah v0.8h, v0.8h, v0.8h
// CHECK: sqrdmlah v0.8h, v0.8h, v0.8h

.arch_extension sb
sb
// CHECK: sb

.arch_extension ssbs
msr SSBS, #1
// CHECK: msr SSBS, #1

.arch_extension tme
tstart x0
// CHECK: tstart x0

.arch_extension fprcvt
fcvtns s0, d1
// CHECK: fcvtns s0, d1

.arch_extension f8f16mm
fmmla v1.8h, v2.16b, v3.16b
// CHECK: fmmla v1.8h, v2.16b, v3.16b

.arch_extension f8f32mm
fmmla v1.4s, v2.16b, v3.16b
// CHECK: fmmla v1.4s, v2.16b, v3.16b
