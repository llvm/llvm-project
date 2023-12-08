// RUN: llvm-mc -triple aarch64-- -mattr=+crc %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=CRC

// RUN: llvm-mc -triple aarch64-- -mcpu=cortex-a55 %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=CRC
// RUN: llvm-mc -triple aarch64-- -mcpu=cortex-a75 %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=CRC
// RUN: llvm-mc -triple aarch64-- -mcpu=tsv110 %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=CRC

// RUN: not llvm-mc -triple aarch64-- %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=NOCRC
// RUN: not llvm-mc -triple aarch64-- -mcpu=cyclone %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=NOCRC

    crc32b w0, w1, w5
    crc32h w3, w5, w6
    crc32w w19, wzr, w20
    crc32x w3, w5, x20

// CRC: crc32b w0, w1, w5
// CRC: crc32h w3, w5, w6
// CRC: crc32w w19, wzr, w20
// CRC: crc32x w3, w5, x20

// NOCRC: error: instruction requires: crc
// NOCRC:     crc32b w0, w1, w5
// NOCRC: error: instruction requires: crc
// NOCRC:     crc32h w3, w5, w6
// NOCRC: error: instruction requires: crc
// NOCRC:     crc32w w19, wzr, w20
// NOCRC: error: instruction requires: crc
// NOCRC:     crc32x w3, w5, x20

    crc32cb w5, w10, w15
    crc32ch w3, w5, w7
    crc32cw w11, w13, w17
    crc32cx w19, w23, x29

// CRC: crc32cb w5, w10, w15
// CRC: crc32ch w3, w5, w7
// CRC: crc32cw w11, w13, w17
// CRC: crc32cx w19, w23, x29

// NOCRC: error: instruction requires: crc
// NOCRC:     crc32cb w5, w10, w15
// NOCRC: error: instruction requires: crc
// NOCRC:     crc32ch w3, w5, w7
// NOCRC: error: instruction requires: crc
// NOCRC:     crc32cw w11, w13, w17
// NOCRC: error: instruction requires: crc
// NOCRC:     crc32cx w19, w23, x29
