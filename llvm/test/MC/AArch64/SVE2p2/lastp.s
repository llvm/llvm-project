// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

lastp   x0, p0, p0.b  // 00100101-00100010-10000000-00000000
// CHECK-INST: lastp   x0, p0, p0.b
// CHECK-ENCODING: [0x00,0x80,0x22,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 25228000 <unknown>

lastp   x23, p11, p13.b  // 00100101-00100010-10101101-10110111
// CHECK-INST: lastp   x23, p11, p13.b
// CHECK-ENCODING: [0xb7,0xad,0x22,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 2522adb7 <unknown>

lastp   xzr, p15, p15.b  // 00100101-00100010-10111101-11111111
// CHECK-INST: lastp   xzr, p15, p15.b
// CHECK-ENCODING: [0xff,0xbd,0x22,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 2522bdff <unknown>

lastp   x0, p0, p0.h  // 00100101-01100010-10000000-00000000
// CHECK-INST: lastp   x0, p0, p0.h
// CHECK-ENCODING: [0x00,0x80,0x62,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 25628000 <unknown>

lastp   x23, p11, p13.h  // 00100101-01100010-10101101-10110111
// CHECK-INST: lastp   x23, p11, p13.h
// CHECK-ENCODING: [0xb7,0xad,0x62,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 2562adb7 <unknown>

lastp   xzr, p15, p15.h  // 00100101-01100010-10111101-11111111
// CHECK-INST: lastp   xzr, p15, p15.h
// CHECK-ENCODING: [0xff,0xbd,0x62,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 2562bdff <unknown>

lastp   x0, p0, p0.s  // 00100101-10100010-10000000-00000000
// CHECK-INST: lastp   x0, p0, p0.s
// CHECK-ENCODING: [0x00,0x80,0xa2,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 25a28000 <unknown>

lastp   x23, p11, p13.s  // 00100101-10100010-10101101-10110111
// CHECK-INST: lastp   x23, p11, p13.s
// CHECK-ENCODING: [0xb7,0xad,0xa2,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 25a2adb7 <unknown>

lastp   xzr, p15, p15.s  // 00100101-10100010-10111101-11111111
// CHECK-INST: lastp   xzr, p15, p15.s
// CHECK-ENCODING: [0xff,0xbd,0xa2,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 25a2bdff <unknown>

lastp   x0, p0, p0.d  // 00100101-11100010-10000000-00000000
// CHECK-INST: lastp   x0, p0, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe2,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 25e28000 <unknown>

lastp   x23, p11, p13.d  // 00100101-11100010-10101101-10110111
// CHECK-INST: lastp   x23, p11, p13.d
// CHECK-ENCODING: [0xb7,0xad,0xe2,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 25e2adb7 <unknown>

lastp   xzr, p15, p15.d  // 00100101-11100010-10111101-11111111
// CHECK-INST: lastp   xzr, p15, p15.d
// CHECK-ENCODING: [0xff,0xbd,0xe2,0x25]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 25e2bdff <unknown>