// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension axp64
// CHECK: error: unknown architectural extension: axp64
// CHECK-NEXT: .arch_extension axp64

.arch_extension nocrc
crc32cx w0, w1, x3
// CHECK: error: instruction requires: crc
// CHECK-NEXT: crc32cx w0, w1, x3

.arch_extension nosm4
sm4e v2.4s, v15.4s
// CHECK: error: instruction requires: sm4
// CHECK-NEXT: sm4e v2.4s, v15.4s

.arch_extension nosha3
sha512h q0, q1, v2.2d
// CHECK: error: instruction requires: sha3
// CHECK-NEXT: sha512h q0, q1, v2.2d

.arch_extension nosha2
sha1h s0, s1
// CHECK: error: instruction requires: sha2
// CHECK-NEXT: sha1h s0, s1

.arch_extension noaes
aese v0.16b, v1.16b
// CHECK: error: instruction requires: aes
// CHECK-NEXT: aese v0.16b, v1.16b

.arch_extension nofp
fminnm d0, d0, d1
// CHECK: error: instruction requires: fp
// CHECK-NEXT: fminnm d0, d0, d1

.arch_extension nosimd
addp v0.4s, v0.4s, v0.4s
// CHECK: error: instruction requires: neon
// CHECK-NEXT: addp v0.4s, v0.4s, v0.4s

.arch_extension noras
esb
// CHECK: error: instruction requires: ras
// CHECK-NEXT: esb

.arch_extension nolse
casa w5, w7, [x20]
// CHECK: error: instruction requires: lse
// CHECK-NEXT: casa w5, w7, [x20]

.arch_extension nopredres
cfp rctx, x0
// CHECK: error: CFPRCTX requires predres
// CHECK-NEXT: cfp rctx, x0

.arch_extension noccdp
dc cvadp, x7
// CHECK: error: DC CVADP requires ccdp
// CHECK-NEXT: dc cvadp, x7

.arch_extension nomte
irg x0, x1
// CHECK: error: instruction requires: mte
// CHECK-NEXT: irg x0, x1

.arch_extension notlb-rmi
tlbi vmalle1os
// CHECK: error: TLBI VMALLE1OS requires tlb-rmi
// CHECK-NEXT: tlbi vmalle1os

.arch_extension nopan-rwv
at s1e1wp, x2
// CHECK: error: AT S1E1WP requires pan-rwv
// CHECK-NEXT: at s1e1wp, x2

.arch_extension noccpp
dc cvap, x7
// CHECK: error: DC CVAP requires ccpp
// CHECK-NEXT: dc cvap, x7

.arch_extension norcpc
ldapr x0, [x1]
// CHECK: error: instruction requires: rcpc
// CHECK-NEXT: ldapr x0, [x1]
