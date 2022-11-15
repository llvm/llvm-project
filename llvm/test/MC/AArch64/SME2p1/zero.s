// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2p1 - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=-sme2p1 - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

zero    za.d[w8, 0:1]  // 11000000-00001100-10000000-00000000
// CHECK-INST: zero    za.d[w8, 0:1]
// CHECK-ENCODING: [0x00,0x80,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c8000 <unknown>

zero    za.d[w10, 10:11]  // 11000000-00001100-11000000-00000101
// CHECK-INST: zero    za.d[w10, 10:11]
// CHECK-ENCODING: [0x05,0xc0,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00cc005 <unknown>

zero    za.d[w11, 14:15]  // 11000000-00001100-11100000-00000111
// CHECK-INST: zero    za.d[w11, 14:15]
// CHECK-ENCODING: [0x07,0xe0,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00ce007 <unknown>

zero    za.d[w8, 10:11]  // 11000000-00001100-10000000-00000101
// CHECK-INST: zero    za.d[w8, 10:11]
// CHECK-ENCODING: [0x05,0x80,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c8005 <unknown>

zero    za.d[w8, 2:3]  // 11000000-00001100-10000000-00000001
// CHECK-INST: zero    za.d[w8, 2:3]
// CHECK-ENCODING: [0x01,0x80,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c8001 <unknown>

zero    za.d[w10, 0:1]  // 11000000-00001100-11000000-00000000
// CHECK-INST: zero    za.d[w10, 0:1]
// CHECK-ENCODING: [0x00,0xc0,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00cc000 <unknown>

zero    za.d[w10, 2:3]  // 11000000-00001100-11000000-00000001
// CHECK-INST: zero    za.d[w10, 2:3]
// CHECK-ENCODING: [0x01,0xc0,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00cc001 <unknown>

zero    za.d[w11, 4:5]  // 11000000-00001100-11100000-00000010
// CHECK-INST: zero    za.d[w11, 4:5]
// CHECK-ENCODING: [0x02,0xe0,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00ce002 <unknown>

zero    za.d[w9, 14:15]  // 11000000-00001100-10100000-00000111
// CHECK-INST: zero    za.d[w9, 14:15]
// CHECK-ENCODING: [0x07,0xa0,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00ca007 <unknown>


zero    za.d[w8, 0:3]  // 11000000-00001110-10000000-00000000
// CHECK-INST: zero    za.d[w8, 0:3]
// CHECK-ENCODING: [0x00,0x80,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e8000 <unknown>

zero    za.d[w10, 4:7]  // 11000000-00001110-11000000-00000001
// CHECK-INST: zero    za.d[w10, 4:7]
// CHECK-ENCODING: [0x01,0xc0,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00ec001 <unknown>

zero    za.d[w11, 12:15]  // 11000000-00001110-11100000-00000011
// CHECK-INST: zero    za.d[w11, 12:15]
// CHECK-ENCODING: [0x03,0xe0,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00ee003 <unknown>

zero    za.d[w8, 4:7]  // 11000000-00001110-10000000-00000001
// CHECK-INST: zero    za.d[w8, 4:7]
// CHECK-ENCODING: [0x01,0x80,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e8001 <unknown>

zero    za.d[w10, 0:3]  // 11000000-00001110-11000000-00000000
// CHECK-INST: zero    za.d[w10, 0:3]
// CHECK-ENCODING: [0x00,0xc0,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00ec000 <unknown>

zero    za.d[w11, 8:11]  // 11000000-00001110-11100000-00000010
// CHECK-INST: zero    za.d[w11, 8:11]
// CHECK-ENCODING: [0x02,0xe0,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00ee002 <unknown>

zero    za.d[w9, 12:15]  // 11000000-00001110-10100000-00000011
// CHECK-INST: zero    za.d[w9, 12:15]
// CHECK-ENCODING: [0x03,0xa0,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00ea003 <unknown>


zero    za.d[w8, 0, vgx2]  // 11000000-00001100-00000000-00000000
// CHECK-INST: zero    za.d[w8, 0, vgx2]
// CHECK-ENCODING: [0x00,0x00,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c0000 <unknown>

zero    za.d[w10, 5, vgx2]  // 11000000-00001100-01000000-00000101
// CHECK-INST: zero    za.d[w10, 5, vgx2]
// CHECK-ENCODING: [0x05,0x40,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c4005 <unknown>

zero    za.d[w11, 7, vgx2]  // 11000000-00001100-01100000-00000111
// CHECK-INST: zero    za.d[w11, 7, vgx2]
// CHECK-ENCODING: [0x07,0x60,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c6007 <unknown>

zero    za.d[w8, 5, vgx2]  // 11000000-00001100-00000000-00000101
// CHECK-INST: zero    za.d[w8, 5, vgx2]
// CHECK-ENCODING: [0x05,0x00,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c0005 <unknown>

zero    za.d[w8, 1, vgx2]  // 11000000-00001100-00000000-00000001
// CHECK-INST: zero    za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x01,0x00,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c0001 <unknown>

zero    za.d[w10, 0, vgx2]  // 11000000-00001100-01000000-00000000
// CHECK-INST: zero    za.d[w10, 0, vgx2]
// CHECK-ENCODING: [0x00,0x40,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c4000 <unknown>

zero    za.d[w10, 1, vgx2]  // 11000000-00001100-01000000-00000001
// CHECK-INST: zero    za.d[w10, 1, vgx2]
// CHECK-ENCODING: [0x01,0x40,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c4001 <unknown>

zero    za.d[w11, 2, vgx2]  // 11000000-00001100-01100000-00000010
// CHECK-INST: zero    za.d[w11, 2, vgx2]
// CHECK-ENCODING: [0x02,0x60,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c6002 <unknown>

zero    za.d[w9, 7, vgx2]  // 11000000-00001100-00100000-00000111
// CHECK-INST: zero    za.d[w9, 7, vgx2]
// CHECK-ENCODING: [0x07,0x20,0x0c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00c2007 <unknown>


zero    za.d[w8, 0:1, vgx2]  // 11000000-00001101-00000000-00000000
// CHECK-INST: zero    za.d[w8, 0:1, vgx2]
// CHECK-ENCODING: [0x00,0x00,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d0000 <unknown>

zero    za.d[w10, 2:3, vgx2]  // 11000000-00001101-01000000-00000001
// CHECK-INST: zero    za.d[w10, 2:3, vgx2]
// CHECK-ENCODING: [0x01,0x40,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d4001 <unknown>

zero    za.d[w11, 6:7, vgx2]  // 11000000-00001101-01100000-00000011
// CHECK-INST: zero    za.d[w11, 6:7, vgx2]
// CHECK-ENCODING: [0x03,0x60,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d6003 <unknown>

zero    za.d[w8, 2:3, vgx2]  // 11000000-00001101-00000000-00000001
// CHECK-INST: zero    za.d[w8, 2:3, vgx2]
// CHECK-ENCODING: [0x01,0x00,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d0001 <unknown>

zero    za.d[w10, 0:1, vgx2]  // 11000000-00001101-01000000-00000000
// CHECK-INST: zero    za.d[w10, 0:1, vgx2]
// CHECK-ENCODING: [0x00,0x40,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d4000 <unknown>

zero    za.d[w11, 4:5, vgx2]  // 11000000-00001101-01100000-00000010
// CHECK-INST: zero    za.d[w11, 4:5, vgx2]
// CHECK-ENCODING: [0x02,0x60,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d6002 <unknown>

zero    za.d[w9, 6:7, vgx2]  // 11000000-00001101-00100000-00000011
// CHECK-INST: zero    za.d[w9, 6:7, vgx2]
// CHECK-ENCODING: [0x03,0x20,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d2003 <unknown>


zero    za.d[w8, 0:3, vgx2]  // 11000000-00001111-00000000-00000000
// CHECK-INST: zero    za.d[w8, 0:3, vgx2]
// CHECK-ENCODING: [0x00,0x00,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f0000 <unknown>

zero    za.d[w10, 4:7, vgx2]  // 11000000-00001111-01000000-00000001
// CHECK-INST: zero    za.d[w10, 4:7, vgx2]
// CHECK-ENCODING: [0x01,0x40,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f4001 <unknown>

zero    za.d[w11, 4:7, vgx2]  // 11000000-00001111-01100000-00000001
// CHECK-INST: zero    za.d[w11, 4:7, vgx2]
// CHECK-ENCODING: [0x01,0x60,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f6001 <unknown>

zero    za.d[w8, 4:7, vgx2]  // 11000000-00001111-00000000-00000001
// CHECK-INST: zero    za.d[w8, 4:7, vgx2]
// CHECK-ENCODING: [0x01,0x00,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f0001 <unknown>

zero    za.d[w10, 0:3, vgx2]  // 11000000-00001111-01000000-00000000
// CHECK-INST: zero    za.d[w10, 0:3, vgx2]
// CHECK-ENCODING: [0x00,0x40,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f4000 <unknown>

zero    za.d[w11, 0:3, vgx2]  // 11000000-00001111-01100000-00000000
// CHECK-INST: zero    za.d[w11, 0:3, vgx2]
// CHECK-ENCODING: [0x00,0x60,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f6000 <unknown>

zero    za.d[w9, 4:7, vgx2]  // 11000000-00001111-00100000-00000001
// CHECK-INST: zero    za.d[w9, 4:7, vgx2]
// CHECK-ENCODING: [0x01,0x20,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f2001 <unknown>


zero    za.d[w8, 0, vgx4]  // 11000000-00001110-00000000-00000000
// CHECK-INST: zero    za.d[w8, 0, vgx4]
// CHECK-ENCODING: [0x00,0x00,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e0000 <unknown>

zero    za.d[w10, 5, vgx4]  // 11000000-00001110-01000000-00000101
// CHECK-INST: zero    za.d[w10, 5, vgx4]
// CHECK-ENCODING: [0x05,0x40,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e4005 <unknown>

zero    za.d[w11, 7, vgx4]  // 11000000-00001110-01100000-00000111
// CHECK-INST: zero    za.d[w11, 7, vgx4]
// CHECK-ENCODING: [0x07,0x60,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e6007 <unknown>

zero    za.d[w8, 5, vgx4]  // 11000000-00001110-00000000-00000101
// CHECK-INST: zero    za.d[w8, 5, vgx4]
// CHECK-ENCODING: [0x05,0x00,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e0005 <unknown>

zero    za.d[w8, 1, vgx4]  // 11000000-00001110-00000000-00000001
// CHECK-INST: zero    za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x01,0x00,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e0001 <unknown>

zero    za.d[w10, 0, vgx4]  // 11000000-00001110-01000000-00000000
// CHECK-INST: zero    za.d[w10, 0, vgx4]
// CHECK-ENCODING: [0x00,0x40,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e4000 <unknown>

zero    za.d[w10, 1, vgx4]  // 11000000-00001110-01000000-00000001
// CHECK-INST: zero    za.d[w10, 1, vgx4]
// CHECK-ENCODING: [0x01,0x40,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e4001 <unknown>

zero    za.d[w11, 2, vgx4]  // 11000000-00001110-01100000-00000010
// CHECK-INST: zero    za.d[w11, 2, vgx4]
// CHECK-ENCODING: [0x02,0x60,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e6002 <unknown>

zero    za.d[w9, 7, vgx4]  // 11000000-00001110-00100000-00000111
// CHECK-INST: zero    za.d[w9, 7, vgx4]
// CHECK-ENCODING: [0x07,0x20,0x0e,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00e2007 <unknown>


zero    za.d[w8, 0:1, vgx4]  // 11000000-00001101-10000000-00000000
// CHECK-INST: zero    za.d[w8, 0:1, vgx4]
// CHECK-ENCODING: [0x00,0x80,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d8000 <unknown>

zero    za.d[w10, 2:3, vgx4]  // 11000000-00001101-11000000-00000001
// CHECK-INST: zero    za.d[w10, 2:3, vgx4]
// CHECK-ENCODING: [0x01,0xc0,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00dc001 <unknown>

zero    za.d[w11, 6:7, vgx4]  // 11000000-00001101-11100000-00000011
// CHECK-INST: zero    za.d[w11, 6:7, vgx4]
// CHECK-ENCODING: [0x03,0xe0,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00de003 <unknown>

zero    za.d[w8, 2:3, vgx4]  // 11000000-00001101-10000000-00000001
// CHECK-INST: zero    za.d[w8, 2:3, vgx4]
// CHECK-ENCODING: [0x01,0x80,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00d8001 <unknown>

zero    za.d[w10, 0:1, vgx4]  // 11000000-00001101-11000000-00000000
// CHECK-INST: zero    za.d[w10, 0:1, vgx4]
// CHECK-ENCODING: [0x00,0xc0,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00dc000 <unknown>

zero    za.d[w11, 4:5, vgx4]  // 11000000-00001101-11100000-00000010
// CHECK-INST: zero    za.d[w11, 4:5, vgx4]
// CHECK-ENCODING: [0x02,0xe0,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00de002 <unknown>

zero    za.d[w9, 6:7, vgx4]  // 11000000-00001101-10100000-00000011
// CHECK-INST: zero    za.d[w9, 6:7, vgx4]
// CHECK-ENCODING: [0x03,0xa0,0x0d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00da003 <unknown>


zero    za.d[w8, 0:3, vgx4]  // 11000000-00001111-10000000-00000000
// CHECK-INST: zero    za.d[w8, 0:3, vgx4]
// CHECK-ENCODING: [0x00,0x80,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f8000 <unknown>

zero    za.d[w10, 4:7, vgx4]  // 11000000-00001111-11000000-00000001
// CHECK-INST: zero    za.d[w10, 4:7, vgx4]
// CHECK-ENCODING: [0x01,0xc0,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00fc001 <unknown>

zero    za.d[w11, 4:7, vgx4]  // 11000000-00001111-11100000-00000001
// CHECK-INST: zero    za.d[w11, 4:7, vgx4]
// CHECK-ENCODING: [0x01,0xe0,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00fe001 <unknown>

zero    za.d[w8, 4:7, vgx4]  // 11000000-00001111-10000000-00000001
// CHECK-INST: zero    za.d[w8, 4:7, vgx4]
// CHECK-ENCODING: [0x01,0x80,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00f8001 <unknown>

zero    za.d[w10, 0:3, vgx4]  // 11000000-00001111-11000000-00000000
// CHECK-INST: zero    za.d[w10, 0:3, vgx4]
// CHECK-ENCODING: [0x00,0xc0,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00fc000 <unknown>

zero    za.d[w11, 0:3, vgx4]  // 11000000-00001111-11100000-00000000
// CHECK-INST: zero    za.d[w11, 0:3, vgx4]
// CHECK-ENCODING: [0x00,0xe0,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00fe000 <unknown>

zero    za.d[w9, 4:7, vgx4]  // 11000000-00001111-10100000-00000001
// CHECK-INST: zero    za.d[w9, 4:7, vgx4]
// CHECK-ENCODING: [0x01,0xa0,0x0f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00fa001 <unknown>

