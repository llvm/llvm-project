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

// -------------------------------------------------------------
// Floating-point convert, narrow and interleave to signed integer, rounding toward zero

fcvtzsn z0.b, { z0.h, z1.h }
// CHECK-INST: fcvtzsn z0.b, { z0.h, z1.h }
// CHECK-ENCODING: encoding: [0x00,0x30,0x4d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654d3000 <unknown>

fcvtzsn z31.b, { z0.h, z1.h }
// CHECK-INST: fcvtzsn z31.b, { z0.h, z1.h }
// CHECK-ENCODING: encoding: [0x1f,0x30,0x4d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654d301f <unknown>

fcvtzsn z0.b, { z30.h, z31.h }
// CHECK-INST: fcvtzsn z0.b, { z30.h, z31.h }
// CHECK-ENCODING: encoding: [0xc0,0x33,0x4d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654d33c0 <unknown>

fcvtzsn z31.b, { z30.h, z31.h }
// CHECK-INST: fcvtzsn z31.b, { z30.h, z31.h }
// CHECK-ENCODING: encoding: [0xdf,0x33,0x4d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654d33df <unknown>

fcvtzsn z0.h, { z0.s, z1.s }
// CHECK-INST: fcvtzsn z0.h, { z0.s, z1.s }
// CHECK-ENCODING: encoding: [0x00,0x30,0x8d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658d3000 <unknown>

fcvtzsn z31.h, { z0.s, z1.s }
// CHECK-INST: fcvtzsn z31.h, { z0.s, z1.s }
// CHECK-ENCODING: encoding: [0x1f,0x30,0x8d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658d301f <unknown>

fcvtzsn z0.h, { z30.s, z31.s }
// CHECK-INST: fcvtzsn z0.h, { z30.s, z31.s }
// CHECK-ENCODING: encoding: [0xc0,0x33,0x8d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658d33c0 <unknown>

fcvtzsn z31.h, { z30.s, z31.s }
// CHECK-INST: fcvtzsn z31.h, { z30.s, z31.s }
// CHECK-ENCODING: encoding: [0xdf,0x33,0x8d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658d33df <unknown>

fcvtzsn z0.s, { z0.d, z1.d }
// CHECK-INST: fcvtzsn z0.s, { z0.d, z1.d }
// CHECK-ENCODING: encoding: [0x00,0x30,0xcd,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cd3000 <unknown>

fcvtzsn z31.s, { z0.d, z1.d }
// CHECK-INST: fcvtzsn z31.s, { z0.d, z1.d }
// CHECK-ENCODING: encoding: [0x1f,0x30,0xcd,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cd301f <unknown>

fcvtzsn z0.s, { z30.d, z31.d }
// CHECK-INST: fcvtzsn z0.s, { z30.d, z31.d }
// CHECK-ENCODING: encoding: [0xc0,0x33,0xcd,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cd33c0 <unknown>

fcvtzsn z31.s, { z30.d, z31.d }
// CHECK-INST: fcvtzsn z31.s, { z30.d, z31.d }
// CHECK-ENCODING: encoding: [0xdf,0x33,0xcd,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cd33df <unknown>

// -------------------------------------------------------------
// Floating-point convert, narrow and interleave to unsigned integer, rounding toward zero

fcvtzun z0.b, { z0.h, z1.h }
// CHECK-INST: fcvtzun z0.b, { z0.h, z1.h }
// CHECK-ENCODING: encoding: [0x00,0x34,0x4d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654d3400 <unknown>

fcvtzun z31.b, { z0.h, z1.h }
// CHECK-INST: fcvtzun z31.b, { z0.h, z1.h }
// CHECK-ENCODING: encoding: [0x1f,0x34,0x4d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654d341f <unknown>

fcvtzun z0.b, { z30.h, z31.h }
// CHECK-INST: fcvtzun z0.b, { z30.h, z31.h }
// CHECK-ENCODING: encoding: [0xc0,0x37,0x4d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654d37c0 <unknown>

fcvtzun z31.b, { z30.h, z31.h }
// CHECK-INST: fcvtzun z31.b, { z30.h, z31.h }
// CHECK-ENCODING: encoding: [0xdf,0x37,0x4d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 654d37df <unknown>

fcvtzun z0.h, { z0.s, z1.s }
// CHECK-INST: fcvtzun z0.h, { z0.s, z1.s }
// CHECK-ENCODING: encoding: [0x00,0x34,0x8d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658d3400 <unknown>

fcvtzun z31.h, { z0.s, z1.s }
// CHECK-INST: fcvtzun z31.h, { z0.s, z1.s }
// CHECK-ENCODING: encoding: [0x1f,0x34,0x8d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658d341f <unknown>

fcvtzun z0.h, { z30.s, z31.s }
// CHECK-INST: fcvtzun z0.h, { z30.s, z31.s }
// CHECK-ENCODING: encoding: [0xc0,0x37,0x8d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658d37c0 <unknown>

fcvtzun z31.h, { z30.s, z31.s }
// CHECK-INST: fcvtzun z31.h, { z30.s, z31.s }
// CHECK-ENCODING: encoding: [0xdf,0x37,0x8d,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 658d37df <unknown>

fcvtzun z0.s, { z0.d, z1.d }
// CHECK-INST: fcvtzun z0.s, { z0.d, z1.d }
// CHECK-ENCODING: encoding: [0x00,0x34,0xcd,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cd3400 <unknown>

fcvtzun z31.s, { z0.d, z1.d }
// CHECK-INST: fcvtzun z31.s, { z0.d, z1.d }
// CHECK-ENCODING: encoding: [0x1f,0x34,0xcd,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cd341f <unknown>

fcvtzun z0.s, { z30.d, z31.d }
// CHECK-INST: fcvtzun z0.s, { z30.d, z31.d }
// CHECK-ENCODING: encoding: [0xc0,0x37,0xcd,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cd37c0 <unknown>

fcvtzun z31.s, { z30.d, z31.d }
// CHECK-INST: fcvtzun z31.s, { z30.d, z31.d }
// CHECK-ENCODING: encoding: [0xdf,0x37,0xcd,0x65]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 65cd37df <unknown>

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
