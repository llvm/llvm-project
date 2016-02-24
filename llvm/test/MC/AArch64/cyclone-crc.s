// RUN: not llvm-mc -triple arm64-apple-ios -mcpu=cyclone %s 2>&1 | FileCheck %s

    crc32b w0, w1, w5
    crc32h w3, w5, w6
    crc32w w19, wzr, w20
    crc32x w3, w5, x20
CHECK: error: instruction requires: crc
CHECK:     crc32b w0, w1, w5
CHECK: error: instruction requires: crc
CHECK:     crc32h w3, w5, w6
CHECK: error: instruction requires: crc
CHECK:     crc32w w19, wzr, w20
CHECK: error: instruction requires: crc
CHECK:     crc32x w3, w5, x20

    crc32cb w5, w10, w15
    crc32ch w3, w5, w7
    crc32cw w11, w13, w17
    crc32cx w19, w23, x29
CHECK: error: instruction requires: crc
CHECK:     crc32cb w5, w10, w15
CHECK: error: instruction requires: crc
CHECK:     crc32ch w3, w5, w7
CHECK: error: instruction requires: crc
CHECK:     crc32cw w11, w13, w17
CHECK: error: instruction requires: crc
CHECK:     crc32cx w19, w23, x29
