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

// -----------------------------------------------------------------------
// Unsigned integer convert to floating-point (bottom, unpredicated)

ucvtf z0.h, z0.b
// CHECK-INST: ucvtf z0.h, z0.b
// CHECK-ENCODING: encoding: [0x00,0x34,0x4c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654c3400 <unknown>

ucvtf z31.h, z31.b
// CHECK-INST: ucvtf z31.h, z31.b
// CHECK-ENCODING: encoding: [0xff,0x37,0x4c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654c37ff <unknown>

ucvtf z0.s, z0.h
// CHECK-INST: ucvtf z0.s, z0.h
// CHECK-ENCODING: encoding: [0x00,0x34,0x8c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658c3400 <unknown>

ucvtf z31.s, z31.h
// CHECK-INST: ucvtf z31.s, z31.h
// CHECK-ENCODING: encoding: [0xff,0x37,0x8c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658c37ff <unknown>

ucvtf z0.d, z0.s
// CHECK-INST: ucvtf z0.d, z0.s
// CHECK-ENCODING: encoding: [0x00,0x34,0xcc,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cc3400 <unknown>

ucvtf z31.d, z31.s
// CHECK-INST: ucvtf z31.d, z31.s
// CHECK-ENCODING: encoding: [0xff,0x37,0xcc,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cc37ff <unknown>

// -----------------------------------------------------------------------
// Unsigned integer convert to floating-point (top, unpredicated)

ucvtflt z0.h, z0.b
// CHECK-INST: ucvtflt z0.h, z0.b
// CHECK-ENCODING: encoding: [0x00,0x3c,0x4c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654c3c00 <unknown>

ucvtflt z31.h, z31.b
// CHECK-INST: ucvtflt z31.h, z31.b
// CHECK-ENCODING: encoding: [0xff,0x3f,0x4c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654c3fff <unknown>

ucvtflt z0.s, z0.h
// CHECK-INST: ucvtflt z0.s, z0.h
// CHECK-ENCODING: encoding: [0x00,0x3c,0x8c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658c3c00 <unknown>

ucvtflt z31.s, z31.h
// CHECK-INST: ucvtflt z31.s, z31.h
// CHECK-ENCODING: encoding: [0xff,0x3f,0x8c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658c3fff <unknown>

ucvtflt z0.d, z0.s
// CHECK-INST: ucvtflt z0.d, z0.s
// CHECK-ENCODING: encoding: [0x00,0x3c,0xcc,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cc3c00 <unknown>

ucvtflt z31.d, z31.s
// CHECK-INST: ucvtflt z31.d, z31.s
// CHECK-ENCODING: encoding: [0xff,0x3f,0xcc,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cc3fff <unknown>
