// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2,+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

sumop4a za0.d, z0.h, z16.h  // 10100000-11100000-00000000-00001000
// CHECK-INST: sumop4a za0.d, z0.h, z16.h
// CHECK-ENCODING: [0x08,0x00,0xe0,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0e00008 <unknown>

sumop4a za5.d, z10.h, z20.h  // 10100000-11100100-00000001-01001101
// CHECK-INST: sumop4a za5.d, z10.h, z20.h
// CHECK-ENCODING: [0x4d,0x01,0xe4,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0e4014d <unknown>

sumop4a za7.d, z14.h, z30.h  // 10100000-11101110-00000001-11001111
// CHECK-INST: sumop4a za7.d, z14.h, z30.h
// CHECK-ENCODING: [0xcf,0x01,0xee,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0ee01cf <unknown>

sumop4a za0.d, z0.h, {z16.h-z17.h}  // 10100000-11110000-00000000-00001000
// CHECK-INST: sumop4a za0.d, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x00,0xf0,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0f00008 <unknown>

sumop4a za5.d, z10.h, {z20.h-z21.h}  // 10100000-11110100-00000001-01001101
// CHECK-INST: sumop4a za5.d, z10.h, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x01,0xf4,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0f4014d <unknown>

sumop4a za7.d, z14.h, {z30.h-z31.h}  // 10100000-11111110-00000001-11001111
// CHECK-INST: sumop4a za7.d, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x01,0xfe,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0fe01cf <unknown>

sumop4a za0.d, {z0.h-z1.h}, z16.h  // 10100000-11100000-00000010-00001000
// CHECK-INST: sumop4a za0.d, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x08,0x02,0xe0,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0e00208 <unknown>

sumop4a za5.d, {z10.h-z11.h}, z20.h  // 10100000-11100100-00000011-01001101
// CHECK-INST: sumop4a za5.d, { z10.h, z11.h }, z20.h
// CHECK-ENCODING: [0x4d,0x03,0xe4,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0e4034d <unknown>

sumop4a za7.d, {z14.h-z15.h}, z30.h  // 10100000-11101110-00000011-11001111
// CHECK-INST: sumop4a za7.d, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xcf,0x03,0xee,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0ee03cf <unknown>

sumop4a za0.d, {z0.h-z1.h}, {z16.h-z17.h}  // 10100000-11110000-00000010-00001000
// CHECK-INST: sumop4a za0.d, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x02,0xf0,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0f00208 <unknown>

sumop4a za5.d, {z10.h-z11.h}, {z20.h-z21.h}  // 10100000-11110100-00000011-01001101
// CHECK-INST: sumop4a za5.d, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x03,0xf4,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0f4034d <unknown>

sumop4a za7.d, {z14.h-z15.h}, {z30.h-z31.h}  // 10100000-11111110-00000011-11001111
// CHECK-INST: sumop4a za7.d, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x03,0xfe,0xa0]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a0fe03cf <unknown>
