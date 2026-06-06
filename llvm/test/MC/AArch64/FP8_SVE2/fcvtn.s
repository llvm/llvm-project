// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+fp8 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+fp8 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2,+fp8 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+fp8 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+fp8 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+fp8 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//
// FCVTN instructions
//
fcvtn   z0.b, {z0.h, z1.h}  // 01100101-00001010-00110000-00000000
// CHECK-INST: fcvtn   z0.b, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x30,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3000 <unknown>

fcvtn   z0.b, {z30.h, z31.h}  // 01100101-00001010-00110011-11000000
// CHECK-INST: fcvtn   z0.b, { z30.h, z31.h }
// CHECK-ENCODING: [0xc0,0x33,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a33c0 <unknown>

fcvtn   z31.b, {z0.h, z1.h}  // 01100101-00001010-00110000-00011111
// CHECK-INST: fcvtn   z31.b, { z0.h, z1.h }
// CHECK-ENCODING: [0x1f,0x30,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a301f <unknown>

fcvtn   z31.b, {z30.h, z31.h}  // 01100101-00001010-00110011-11011111
// CHECK-INST: fcvtn   z31.b, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0x33,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a33df <unknown>

//
// FCVTNB instructions
//
fcvtnb  z0.b, {z0.s, z1.s}  // 01100101-00001010-00110100-00000000
// CHECK-INST: fcvtnb  z0.b, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x34,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3400 <unknown>

fcvtnb  z0.b, {z30.s, z31.s}  // 01100101-00001010-00110111-11000000
// CHECK-INST: fcvtnb  z0.b, { z30.s, z31.s }
// CHECK-ENCODING: [0xc0,0x37,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a37c0 <unknown>

fcvtnb  z31.b, {z0.s, z1.s}  // 01100101-00001010-00110100-00011111
// CHECK-INST: fcvtnb  z31.b, { z0.s, z1.s }
// CHECK-ENCODING: [0x1f,0x34,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a341f <unknown>

fcvtnb  z31.b, {z30.s, z31.s}  // 01100101-00001010-00110111-11011111
// CHECK-INST: fcvtnb  z31.b, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0x37,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a37df <unknown>


//
// BFCVTN instructions
//
bfcvtn  z0.b, {z0.h, z1.h}  // 01100101-00001010-00111000-00000000
// CHECK-INST: bfcvtn  z0.b, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x38,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3800 <unknown>

bfcvtn  z0.b, {z30.h, z31.h}  // 01100101-00001010-00111011-11000000
// CHECK-INST: bfcvtn  z0.b, { z30.h, z31.h }
// CHECK-ENCODING: [0xc0,0x3b,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3bc0 <unknown>

bfcvtn  z31.b, {z0.h, z1.h}  // 01100101-00001010-00111000-00011111
// CHECK-INST: bfcvtn  z31.b, { z0.h, z1.h }
// CHECK-ENCODING: [0x1f,0x38,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a381f <unknown>

bfcvtn  z31.b, {z30.h, z31.h}  // 01100101-00001010-00111011-11011111
// CHECK-INST: bfcvtn  z31.b, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0x3b,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3bdf <unknown>


//
// FCVTNT instructions
//
fcvtnt  z0.b, {z0.s, z1.s}  // 01100101-00001010-00111100-00000000
// CHECK-INST: fcvtnt  z0.b, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x3c,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3c00 <unknown>

fcvtnt  z0.b, {z30.s, z31.s}  // 01100101-00001010-00111111-11000000
// CHECK-INST: fcvtnt  z0.b, { z30.s, z31.s }
// CHECK-ENCODING: [0xc0,0x3f,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3fc0 <unknown>

fcvtnt  z31.b, {z0.s, z1.s}  // 01100101-00001010-00111100-00011111
// CHECK-INST: fcvtnt  z31.b, { z0.s, z1.s }
// CHECK-ENCODING: [0x1f,0x3c,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3c1f <unknown>

fcvtnt  z31.b, {z30.s, z31.s}  // 01100101-00001010-00111111-11011111
// CHECK-INST: fcvtnt  z31.b, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0x3f,0x0a,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650a3fdf <unknown>
