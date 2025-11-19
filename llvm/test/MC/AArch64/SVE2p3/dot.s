// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p3 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2p3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p3 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

sdot z0.h, z0.b, z0.b
// CHECK-INST: sdot z0.h, z0.b, z0.b
// CHECK-ENCODING: encoding: [0x00,0x00,0x40,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44400000 <unknown>

sdot z10.h, z10.b, z10.b
// CHECK-INST: sdot z10.h, z10.b, z10.b
// CHECK-ENCODING: encoding: [0x4a,0x01,0x4a,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 444a014a <unknown>

sdot z21.h, z21.b, z21.b
// CHECK-INST: sdot z21.h, z21.b, z21.b
// CHECK-ENCODING: encoding: [0xb5,0x02,0x55,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 445502b5 <unknown>

sdot z31.h, z31.b, z31.b
// CHECK-INST: sdot z31.h, z31.b, z31.b
// CHECK-ENCODING: encoding: [0xff,0x03,0x5f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 445f03ff <unknown>

movprfx z0, z7
sdot z0.h, z1.b, z2.b
// CHECK-INST: movprfx z0, z7
// CHECK-INST: sdot z0.h, z1.b, z2.b
// CHECK-ENCODING: encoding: [0x20,0x00,0x42,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44420020 <unknown>

// sdot indexed

sdot z0.h, z0.b, z0.b[0]
// CHECK-INST: sdot z0.h, z0.b, z0.b[0]
// CHECK-ENCODING: encoding: [0x00,0x00,0x20,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44200000 <unknown>

sdot z31.h, z31.b, z7.b[0]
// CHECK-INST: sdot z31.h, z31.b, z7.b[0]
// CHECK-ENCODING: encoding: [0xff,0x03,0x27,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 442703ff <unknown>

sdot z0.h, z0.b, z0.b[1]
// CHECK-INST: sdot z0.h, z0.b, z0.b[1]
// CHECK-ENCODING: encoding: [0x00,0x00,0x28,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44280000 <unknown>

sdot z31.h, z31.b, z7.b[1]
// CHECK-INST: sdot z31.h, z31.b, z7.b[1]
// CHECK-ENCODING: encoding: [0xff,0x03,0x2f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 442f03ff <unknown>

sdot z0.h, z0.b, z0.b[7]
// CHECK-INST: sdot z0.h, z0.b, z0.b[7]
// CHECK-ENCODING: encoding: [0x00,0x00,0x78,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44780000 <unknown>

sdot z31.h, z31.b, z7.b[7]
// CHECK-INST: sdot z31.h, z31.b, z7.b[7]
// CHECK-ENCODING: encoding: [0xff,0x03,0x7f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 447f03ff <unknown>

movprfx z0, z7
sdot z0.h, z1.b, z2.b[0]
// CHECK-INST: movprfx z0, z7
// CHECK-INST: sdot z0.h, z1.b, z2.b[0]
// CHECK-ENCODING: encoding: [0x20,0x00,0x22,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44220020 <unknown>

// udot

udot z0.h, z0.b, z0.b
// CHECK-INST: udot z0.h, z0.b, z0.b
// CHECK-ENCODING: encoding: [0x00,0x04,0x40,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44400400 <unknown>

udot z10.h, z10.b, z10.b
// CHECK-INST: udot z10.h, z10.b, z10.b
// CHECK-ENCODING: encoding: [0x4a,0x05,0x4a,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 444a054a <unknown>

udot z21.h, z21.b, z21.b
// CHECK-INST: udot z21.h, z21.b, z21.b
// CHECK-ENCODING: encoding: [0xb5,0x06,0x55,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 445506b5 <unknown>

udot z31.h, z31.b, z31.b
// CHECK-INST: udot z31.h, z31.b, z31.b
// CHECK-ENCODING: encoding: [0xff,0x07,0x5f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 445f07ff <unknown>

movprfx z0, z7
udot z0.h, z1.b, z2.b
// CHECK-INST: movprfx z0, z7
// CHECK-INST: udot z0.h, z1.b, z2.b
// CHECK-ENCODING: encoding: [0x20,0x04,0x42,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44420420 <unknown>

// udot indexed

udot z0.h, z0.b, z0.b[0]
// CHECK-INST: udot z0.h, z0.b, z0.b[0]
// CHECK-ENCODING: encoding: [0x00,0x04,0x20,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44200400 <unknown>

udot z31.h, z31.b, z7.b[0]
// CHECK-INST: udot z31.h, z31.b, z7.b[0]
// CHECK-ENCODING: encoding: [0xff,0x07,0x27,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 442707ff <unknown>

udot z0.h, z0.b, z0.b[1]
// CHECK-INST: udot z0.h, z0.b, z0.b[1]
// CHECK-ENCODING: encoding: [0x00,0x04,0x28,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44280400 <unknown>

udot z31.h, z31.b, z7.b[1]
// CHECK-INST: udot z31.h, z31.b, z7.b[1]
// CHECK-ENCODING: encoding: [0xff,0x07,0x2f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 442f07ff <unknown>

udot z0.h, z0.b, z0.b[7]
// CHECK-INST: udot z0.h, z0.b, z0.b[7]
// CHECK-ENCODING: encoding: [0x00,0x04,0x78,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44780400 <unknown>

udot z31.h, z31.b, z7.b[7]
// CHECK-INST: udot z31.h, z31.b, z7.b[7]
// CHECK-ENCODING: encoding: [0xff,0x07,0x7f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 447f07ff <unknown>

movprfx z0, z7
udot z0.h, z1.b, z2.b[0]
// CHECK-INST: movprfx z0, z7
// CHECK-INST: udot z0.h, z1.b, z2.b[0]
// CHECK-ENCODING: encoding: [0x20,0x04,0x22,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44220420 <unknown>
