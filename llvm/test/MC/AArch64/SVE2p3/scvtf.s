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
// Signed integer convert to floating-point (bottom, unpredicated)

scvtf z0.h, z0.b
// CHECK-INST: scvtf z0.h, z0.b
// CHECK-ENCODING: encoding: [0x00,0x30,0x4c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654c3000 <unknown>

scvtf z31.h, z31.b
// CHECK-INST: scvtf z31.h, z31.b
// CHECK-ENCODING: encoding: [0xff,0x33,0x4c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654c33ff <unknown>

scvtf z0.s, z0.h
// CHECK-INST: scvtf z0.s, z0.h
// CHECK-ENCODING: encoding: [0x00,0x30,0x8c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658c3000 <unknown>

scvtf z31.s, z31.h
// CHECK-INST: scvtf z31.s, z31.h
// CHECK-ENCODING: encoding: [0xff,0x33,0x8c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658c33ff <unknown>

scvtf z0.d, z0.s
// CHECK-INST: scvtf z0.d, z0.s
// CHECK-ENCODING: encoding: [0x00,0x30,0xcc,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cc3000 <unknown>

scvtf z31.d, z31.s
// CHECK-INST: scvtf z31.d, z31.s
// CHECK-ENCODING: encoding: [0xff,0x33,0xcc,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cc33ff <unknown>

// -----------------------------------------------------------------------
// Signed integer convert to floating-point (top, unpredicated)

scvtflt z0.h, z0.b
// CHECK-INST: scvtflt z0.h, z0.b
// CHECK-ENCODING: encoding: [0x00,0x38,0x4c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654c3800 <unknown>

scvtflt z31.h, z31.b
// CHECK-INST: scvtflt z31.h, z31.b
// CHECK-ENCODING: encoding: [0xff,0x3b,0x4c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654c3bff <unknown>

scvtflt z0.s, z0.h
// CHECK-INST: scvtflt z0.s, z0.h
// CHECK-ENCODING: encoding: [0x00,0x38,0x8c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658c3800 <unknown>

scvtflt z31.s, z31.h
// CHECK-INST: scvtflt z31.s, z31.h
// CHECK-ENCODING: encoding: [0xff,0x3b,0x8c,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658c3bff <unknown>

scvtflt z0.d, z0.s
// CHECK-INST: scvtflt z0.d, z0.s
// CHECK-ENCODING: encoding: [0x00,0x38,0xcc,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cc3800 <unknown>

scvtflt z31.d, z31.s
// CHECK-INST: scvtflt z31.d, z31.s
// CHECK-ENCODING: encoding: [0xff,0x3b,0xcc,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cc3bff <unknown>
