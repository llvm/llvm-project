// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


pmov    p0.h, z0[0]  // 00000101-00101100-00111000-00000000
// CHECK-INST: pmov    p0.h, z0[0]
// CHECK-ENCODING: [0x00,0x38,0x2c,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052c3800 <unknown>

pmov    p5.h, z10[0]  // 00000101-00101100-00111001-01000101
// CHECK-INST: pmov    p5.h, z10[0]
// CHECK-ENCODING: [0x45,0x39,0x2c,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052c3945 <unknown>

pmov    p7.h, z13[0]  // 00000101-00101100-00111001-10100111
// CHECK-INST: pmov    p7.h, z13[0]
// CHECK-ENCODING: [0xa7,0x39,0x2c,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052c39a7 <unknown>

pmov    p15.h, z31[1]  // 00000101-00101110-00111011-11101111
// CHECK-INST: pmov    p15.h, z31[1]
// CHECK-ENCODING: [0xef,0x3b,0x2e,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052e3bef <unknown>

pmov    p0.s, z0[0]  // 00000101-01101000-00111000-00000000
// CHECK-INST: pmov    p0.s, z0[0]
// CHECK-ENCODING: [0x00,0x38,0x68,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05683800 <unknown>

pmov    p5.s, z10[2]  // 00000101-01101100-00111001-01000101
// CHECK-INST: pmov    p5.s, z10[2]
// CHECK-ENCODING: [0x45,0x39,0x6c,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056c3945 <unknown>

pmov    p7.s, z13[0]  // 00000101-01101000-00111001-10100111
// CHECK-INST: pmov    p7.s, z13[0]
// CHECK-ENCODING: [0xa7,0x39,0x68,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056839a7 <unknown>

pmov    p15.s, z31[3]  // 00000101-01101110-00111011-11101111
// CHECK-INST: pmov    p15.s, z31[3]
// CHECK-ENCODING: [0xef,0x3b,0x6e,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056e3bef <unknown>

pmov    p0.d, z0[0]  // 00000101-10101000-00111000-00000000
// CHECK-INST: pmov    p0.d, z0[0]
// CHECK-ENCODING: [0x00,0x38,0xa8,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05a83800 <unknown>

pmov    p5.d, z10[6]  // 00000101-11101100-00111001-01000101
// CHECK-INST: pmov    p5.d, z10[6]
// CHECK-ENCODING: [0x45,0x39,0xec,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05ec3945 <unknown>

pmov    p7.d, z13[4]  // 00000101-11101000-00111001-10100111
// CHECK-INST: pmov    p7.d, z13[4]
// CHECK-ENCODING: [0xa7,0x39,0xe8,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05e839a7 <unknown>

pmov    p15.d, z31[7]  // 00000101-11101110-00111011-11101111
// CHECK-INST: pmov    p15.d, z31[7]
// CHECK-ENCODING: [0xef,0x3b,0xee,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05ee3bef <unknown>

pmov    p0.b, z0  // 00000101-00101010-00111000-00000000
// CHECK-INST: pmov    p0.b, z0
// CHECK-ENCODING: [0x00,0x38,0x2a,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052a3800 <unknown>

pmov    p5.b, z10  // 00000101-00101010-00111001-01000101
// CHECK-INST: pmov    p5.b, z10
// CHECK-ENCODING: [0x45,0x39,0x2a,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052a3945 <unknown>

pmov    p7.b, z13  // 00000101-00101010-00111001-10100111
// CHECK-INST: pmov    p7.b, z13
// CHECK-ENCODING: [0xa7,0x39,0x2a,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052a39a7 <unknown>

pmov    p15.b, z31  // 00000101-00101010-00111011-11101111
// CHECK-INST: pmov    p15.b, z31
// CHECK-ENCODING: [0xef,0x3b,0x2a,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052a3bef <unknown>

pmov    p0.b, z0[0] // 00000101-00101010-00111000-00000000
// CHECK-INST: pmov    p0.b, z0
// CHECK-ENCODING: [0x00,0x38,0x2a,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052a3800 <unknown>

pmov    z0[0], p0.h  // 00000101-00101101-00111000-00000000
// CHECK-INST: pmov    z0[0], p0.h
// CHECK-ENCODING: [0x00,0x38,0x2d,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052d3800 <unknown>

pmov    z21[0], p10.h  // 00000101-00101101-00111001-01010101
// CHECK-INST: pmov    z21[0], p10.h
// CHECK-ENCODING: [0x55,0x39,0x2d,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052d3955 <unknown>

pmov    z23[0], p13.h  // 00000101-00101101-00111001-10110111
// CHECK-INST: pmov    z23[0], p13.h
// CHECK-ENCODING: [0xb7,0x39,0x2d,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052d39b7 <unknown>

pmov    z31[1], p15.h  // 00000101-00101111-00111001-11111111
// CHECK-INST: pmov    z31[1], p15.h
// CHECK-ENCODING: [0xff,0x39,0x2f,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052f39ff <unknown>


pmov    z0[0], p0.s  // 00000101-01101001-00111000-00000000
// CHECK-INST: pmov    z0[0], p0.s
// CHECK-ENCODING: [0x00,0x38,0x69,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05693800 <unknown>

pmov    z21[2], p10.s  // 00000101-01101101-00111001-01010101
// CHECK-INST: pmov    z21[2], p10.s
// CHECK-ENCODING: [0x55,0x39,0x6d,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056d3955 <unknown>

pmov    z23[0], p13.s  // 00000101-01101001-00111001-10110111
// CHECK-INST: pmov    z23[0], p13.s
// CHECK-ENCODING: [0xb7,0x39,0x69,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056939b7 <unknown>

pmov    z31[3], p15.s  // 00000101-01101111-00111001-11111111
// CHECK-INST: pmov    z31[3], p15.s
// CHECK-ENCODING: [0xff,0x39,0x6f,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056f39ff <unknown>

pmov    z0[0], p0.d  // 00000101-10101001-00111000-00000000
// CHECK-INST: pmov    z0[0], p0.d
// CHECK-ENCODING: [0x00,0x38,0xa9,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05a93800 <unknown>

pmov    z21[6], p10.d  // 00000101-11101101-00111001-01010101
// CHECK-INST: pmov    z21[6], p10.d
// CHECK-ENCODING: [0x55,0x39,0xed,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05ed3955 <unknown>

pmov    z23[4], p13.d  // 00000101-11101001-00111001-10110111
// CHECK-INST: pmov    z23[4], p13.d
// CHECK-ENCODING: [0xb7,0x39,0xe9,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05e939b7 <unknown>

pmov    z31[7], p15.d  // 00000101-11101111-00111001-11111111
// CHECK-INST: pmov    z31[7], p15.d
// CHECK-ENCODING: [0xff,0x39,0xef,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05ef39ff <unknown>

pmov    z0, p0.b  // 00000101-00101011-00111000-00000000
// CHECK-INST: pmov    z0, p0.b
// CHECK-ENCODING: [0x00,0x38,0x2b,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052b3800 <unknown>

pmov    z21, p10.b  // 00000101-00101011-00111001-01010101
// CHECK-INST: pmov    z21, p10.b
// CHECK-ENCODING: [0x55,0x39,0x2b,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052b3955 <unknown>

pmov    z23, p13.b  // 00000101-00101011-00111001-10110111
// CHECK-INST: pmov    z23, p13.b
// CHECK-ENCODING: [0xb7,0x39,0x2b,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052b39b7 <unknown>

pmov    z31, p15.b  // 00000101-00101011-00111001-11111111
// CHECK-INST: pmov    z31, p15.b
// CHECK-ENCODING: [0xff,0x39,0x2b,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052b39ff <unknown>

pmov    z0[0], p0.b  // 00000101-00101011-00111000-00000000
// CHECK-INST: pmov    z0, p0.b
// CHECK-ENCODING: [0x00,0x38,0x2b,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052b3800 <unknown>
