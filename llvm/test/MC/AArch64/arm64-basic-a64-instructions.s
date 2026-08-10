// RUN: llvm-mc -triple arm64 -mattr=+crc -show-encoding < %s | FileCheck %s

        crc32b  w5, w7, w20
        crc32h  w28, wzr, w30
        crc32w  w0, w1, w2
        crc32x  w7, w9, x20
        crc32cb w9, w5, w4
        crc32ch w13, w17, w25
        crc32cw wzr, w3, w5
        crc32cx w18, w16, xzr
// CHECK: crc32b   w5, w7, w20             // encoding: [0xe5,0x40,0xd4,0x1a]
// CHECK: crc32h   w28, wzr, w30           // encoding: [0xfc,0x47,0xde,0x1a]
// CHECK: crc32w   w0, w1, w2              // encoding: [0x20,0x48,0xc2,0x1a]
// CHECK: crc32x   w7, w9, x20             // encoding: [0x27,0x4d,0xd4,0x9a]
// CHECK: crc32cb  w9, w5, w4              // encoding: [0xa9,0x50,0xc4,0x1a]
// CHECK: crc32ch  w13, w17, w25           // encoding: [0x2d,0x56,0xd9,0x1a]
// CHECK: crc32cw  wzr, w3, w5             // encoding: [0x7f,0x58,0xc5,0x1a]
// CHECK: crc32cx  w18, w16, xzr           // encoding: [0x12,0x5e,0xdf,0x9a]
