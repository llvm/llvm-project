// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.cpu generic
fminnm d0, d0, d1
// CHECK: fminnm d0, d0, d1

.cpu generic+fp
fminnm d0, d0, d1
// CHECK: fminnm d0, d0, d1

.cpu generic+simd
addp v0.4s, v0.4s, v0.4s
// CHECK: addp v0.4s, v0.4s, v0.4s

.cpu generic+crc
crc32cx w0, w1, x3
// CHECK: crc32cx w0, w1, x3

.cpu generic+crypto+nocrc
aesd v0.16b, v2.16b
// CHECK: aesd v0.16b, v2.16b

.cpu generic+lse
casa  w5, w7, [x20]
// CHECK: casa  w5, w7, [x20]

.cpu generic+aes
aese v0.16b, v1.16b
// CHECK: aese  v0.16b, v1.16b

.cpu generic+sha2
sha1h s0, s1
// CHECK: sha1h s0, s1

.cpu generic+sha3
sha512h q0, q1, v2.2d
// CHECK: sha512h q0, q1, v2.2d

.cpu generic+sm4
sm4e v2.4s, v15.4s
// CHECK: sm4e  v2.4s, v15.4s
