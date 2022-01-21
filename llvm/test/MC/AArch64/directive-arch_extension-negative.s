// RUN: not llvm-mc -triple aarch64 \
// RUN: -mattr=+crc,+sm4,+sha3,+sha2,+aes,+fp,+neon,+ras,+lse,+predres,+ccdp,+mte,+tlb-rmi,+pan-rwv,+ccpp,+rcpc,+ls64,+flagm,+hbc,+mops \
// RUN: -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension axp64
// CHECK: error: unknown architectural extension: axp64
// CHECK-NEXT: .arch_extension axp64

crc32cx w0, w1, x3
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: crc
.arch_extension nocrc
crc32cx w0, w1, x3
// CHECK: [[@LINE-1]]:1: error: instruction requires: crc
// CHECK-NEXT: crc32cx w0, w1, x3

sm4e v2.4s, v15.4s
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: sm4
.arch_extension nosm4
sm4e v2.4s, v15.4s
// CHECK: [[@LINE-1]]:1: error: instruction requires: sm4
// CHECK-NEXT: sm4e v2.4s, v15.4s

sha512h q0, q1, v2.2d
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: sha3
.arch_extension nosha3
sha512h q0, q1, v2.2d
// CHECK: [[@LINE-1]]:1: error: instruction requires: sha3
// CHECK-NEXT: sha512h q0, q1, v2.2d

sha1h s0, s1
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: sha2
.arch_extension nosha2
sha1h s0, s1
// CHECK: [[@LINE-1]]:1: error: instruction requires: sha2
// CHECK-NEXT: sha1h s0, s1

aese v0.16b, v1.16b
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: aes
.arch_extension noaes
aese v0.16b, v1.16b
// CHECK: [[@LINE-1]]:1: error: instruction requires: aes
// CHECK-NEXT: aese v0.16b, v1.16b

fminnm d0, d0, d1
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: fp
.arch_extension nofp
fminnm d0, d0, d1
// CHECK: [[@LINE-1]]:1: error: instruction requires: fp
// CHECK-NEXT: fminnm d0, d0, d1

addp v0.4s, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: neon
.arch_extension nosimd
addp v0.4s, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:1: error: instruction requires: neon
// CHECK-NEXT: addp v0.4s, v0.4s, v0.4s

esb
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: ras
.arch_extension noras
esb
// CHECK: [[@LINE-1]]:1: error: instruction requires: ras
// CHECK-NEXT: esb

casa w5, w7, [x20]
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: lse
.arch_extension nolse
casa w5, w7, [x20]
// CHECK: [[@LINE-1]]:1: error: instruction requires: lse
// CHECK-NEXT: casa w5, w7, [x20]

cfp rctx, x0
// CHECK-NOT: [[@LINE-1]]:5: error: CFPRCTX requires: predres
.arch_extension nopredres
cfp rctx, x0
// CHECK: [[@LINE-1]]:5: error: CFPRCTX requires: predres
// CHECK-NEXT: cfp rctx, x0

dc cvadp, x7
// CHECK-NOT: [[@LINE-1]]:4: error: DC CVADP requires: ccdp
.arch_extension noccdp
dc cvadp, x7
// CHECK: [[@LINE-1]]:4: error: DC CVADP requires: ccdp
// CHECK-NEXT: dc cvadp, x7

irg x0, x1
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: mte
.arch_extension nomte
irg x0, x1
// CHECK: [[@LINE-1]]:1: error: instruction requires: mte
// CHECK-NEXT: irg x0, x1

tlbi vmalle1os
// CHECK-NOT: [[@LINE-1]]:6: error: TLBI VMALLE1OS requires: tlb-rmi
.arch_extension notlb-rmi
tlbi vmalle1os
// CHECK: [[@LINE-1]]:6: error: TLBI VMALLE1OS requires: tlb-rmi
// CHECK-NEXT: tlbi vmalle1os

at s1e1wp, x2
// CHECK-NOT: [[@LINE-1]]:4: error: AT S1E1WP requires: pan-rwv
.arch_extension nopan-rwv
at s1e1wp, x2
// CHECK: [[@LINE-1]]:4: error: AT S1E1WP requires: pan-rwv
// CHECK-NEXT: at s1e1wp, x2

dc cvap, x7
// CHECK-NOT: [[@LINE-1]]:4: error: DC CVAP requires: ccpp
.arch_extension noccpp
dc cvap, x7
// CHECK: [[@LINE-1]]:4: error: DC CVAP requires: ccpp
// CHECK-NEXT: dc cvap, x7

ldapr x0, [x1]
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: rcpc
.arch_extension norcpc
ldapr x0, [x1]
// CHECK: [[@LINE-1]]:1: error: instruction requires: rcpc
// CHECK-NEXT: ldapr x0, [x1]

ld64b x0, [x13]
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: ls64
.arch_extension nols64
ld64b x0, [x13]
// CHECK: [[@LINE-1]]:1: error: instruction requires: ls64
// CHECK-NEXT: ld64b x0, [x13]

cfinv
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: flagm
.arch_extension noflagm
cfinv
// CHECK: [[@LINE-1]]:1: error: instruction requires: flagm
// CHECK-NEXT: cfinv

lbl:
bc.eq lbl
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: hbc
.arch_extension nohbc
bc.eq lbl
// CHECK: [[@LINE-1]]:1: error: instruction requires: hbc
// CHECK-NEXT: bc.eq lbl

cpyfp [x0]!, [x1]!, x2!
// CHECK-NOT: [[@LINE-1]]:1: error: instruction requires: mops
.arch_extension nomops
cpyfp [x0]!, [x1]!, x2!
// CHECK: [[@LINE-1]]:1: error: instruction requires: mops
// CHECK-NEXT: cpyfp [x0]!, [x1]!, x2!
