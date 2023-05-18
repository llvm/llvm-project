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


zipq1   z0.h, z0.h, z0.h  // 01000100-01000000-11100000-00000000
// CHECK-INST: zipq1   z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xe0,0x40,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4440e000 <unknown>

zipq1   z21.h, z10.h, z21.h  // 01000100-01010101-11100001-01010101
// CHECK-INST: zipq1   z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0xe1,0x55,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4455e155 <unknown>

zipq1   z23.h, z13.h, z8.h  // 01000100-01001000-11100001-10110111
// CHECK-INST: zipq1   z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xe1,0x48,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4448e1b7 <unknown>

zipq1   z31.h, z31.h, z31.h  // 01000100-01011111-11100011-11111111
// CHECK-INST: zipq1   z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0xe3,0x5f,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 445fe3ff <unknown>


zipq1   z0.s, z0.s, z0.s  // 01000100-10000000-11100000-00000000
// CHECK-INST: zipq1   z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0xe0,0x80,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4480e000 <unknown>

zipq1   z21.s, z10.s, z21.s  // 01000100-10010101-11100001-01010101
// CHECK-INST: zipq1   z21.s, z10.s, z21.s
// CHECK-ENCODING: [0x55,0xe1,0x95,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4495e155 <unknown>

zipq1   z23.s, z13.s, z8.s  // 01000100-10001000-11100001-10110111
// CHECK-INST: zipq1   z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0xe1,0x88,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4488e1b7 <unknown>

zipq1   z31.s, z31.s, z31.s  // 01000100-10011111-11100011-11111111
// CHECK-INST: zipq1   z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0xe3,0x9f,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 449fe3ff <unknown>


zipq1   z0.d, z0.d, z0.d  // 01000100-11000000-11100000-00000000
// CHECK-INST: zipq1   z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xe0,0xc0,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 44c0e000 <unknown>

zipq1   z21.d, z10.d, z21.d  // 01000100-11010101-11100001-01010101
// CHECK-INST: zipq1   z21.d, z10.d, z21.d
// CHECK-ENCODING: [0x55,0xe1,0xd5,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 44d5e155 <unknown>

zipq1   z23.d, z13.d, z8.d  // 01000100-11001000-11100001-10110111
// CHECK-INST: zipq1   z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0xe1,0xc8,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 44c8e1b7 <unknown>

zipq1   z31.d, z31.d, z31.d  // 01000100-11011111-11100011-11111111
// CHECK-INST: zipq1   z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0xe3,0xdf,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 44dfe3ff <unknown>


zipq1   z0.b, z0.b, z0.b  // 01000100-00000000-11100000-00000000
// CHECK-INST: zipq1   z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xe0,0x00,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4400e000 <unknown>

zipq1   z21.b, z10.b, z21.b  // 01000100-00010101-11100001-01010101
// CHECK-INST: zipq1   z21.b, z10.b, z21.b
// CHECK-ENCODING: [0x55,0xe1,0x15,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4415e155 <unknown>

zipq1   z23.b, z13.b, z8.b  // 01000100-00001000-11100001-10110111
// CHECK-INST: zipq1   z23.b, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xe1,0x08,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 4408e1b7 <unknown>

zipq1   z31.b, z31.b, z31.b  // 01000100-00011111-11100011-11111111
// CHECK-INST: zipq1   z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0xe3,0x1f,0x44]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 441fe3ff <unknown>

