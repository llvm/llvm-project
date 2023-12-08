// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fp8 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fp8 < %s \
// RUN:        | llvm-objdump -d --mattr=+fp8 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fp8 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fp8 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+fp8 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

///
/// BF1CVTL instructions.
///
bf1cvtl v0.8h, v0.8b
// CHECK-INST: bf1cvtl v0.8h, v0.8b
// CHECK-ENCODING: [0x00,0x78,0xa1,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ea17800 <unknown>

bf1cvtl v0.8h, v31.8b
// CHECK-INST: bf1cvtl v0.8h, v31.8b
// CHECK-ENCODING: [0xe0,0x7b,0xa1,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ea17be0 <unknown>

bf1cvtl v31.8h, v31.8b
// CHECK-INST: bf1cvtl v31.8h, v31.8b
// CHECK-ENCODING: [0xff,0x7b,0xa1,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ea17bff <unknown>

///
/// BF1CVTL2 instructions.
///
bf1cvtl2 v0.8h, v0.16b
// CHECK-INST: bf1cvtl2 v0.8h, v0.16b
// CHECK-ENCODING: [0x00,0x78,0xa1,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ea17800 <unknown>

bf1cvtl2 v0.8h, v31.16b
// CHECK-INST: bf1cvtl2 v0.8h, v31.16b
// CHECK-ENCODING: [0xe0,0x7b,0xa1,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ea17be0 <unknown>

bf1cvtl2 v31.8h, v31.16b
// CHECK-INST: bf1cvtl2 v31.8h, v31.16b
// CHECK-ENCODING: [0xff,0x7b,0xa1,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ea17bff <unknown>

///
/// BF2CVTL instructions.
///
bf2cvtl v0.8h, v0.8b
// CHECK-INST: bf2cvtl v0.8h, v0.8b
// CHECK-ENCODING: [0x00,0x78,0xe1,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ee17800 <unknown>

bf2cvtl v0.8h, v31.8b
// CHECK-INST: bf2cvtl v0.8h, v31.8b
// CHECK-ENCODING: [0xe0,0x7b,0xe1,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ee17be0 <unknown>

bf2cvtl v31.8h, v31.8b
// CHECK-INST: bf2cvtl v31.8h, v31.8b
// CHECK-ENCODING: [0xff,0x7b,0xe1,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ee17bff <unknown>

///
/// BF2CVTL2 instructions.
///
bf2cvtl2 v0.8h, v0.16b
// CHECK-INST: bf2cvtl2 v0.8h, v0.16b
// CHECK-ENCODING: [0x00,0x78,0xe1,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ee17800 <unknown>

bf2cvtl2 v0.8h, v31.16b
// CHECK-INST: bf2cvtl2 v0.8h, v31.16b
// CHECK-ENCODING: [0xe0,0x7b,0xe1,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ee17be0 <unknown>

bf2cvtl2 v31.8h, v31.16b
// CHECK-INST: bf2cvtl2 v31.8h, v31.16b
// CHECK-ENCODING: [0xff,0x7b,0xe1,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ee17bff <unknown>

///
/// F1CVTL instructions.
///
f1cvtl v0.8h, v0.8b
// CHECK-INST: f1cvtl v0.8h, v0.8b
// CHECK-ENCODING: [0x00,0x78,0x21,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2e217800 <unknown>

f1cvtl v0.8h, v31.8b
// CHECK-INST: f1cvtl v0.8h, v31.8b
// CHECK-ENCODING: [0xe0,0x7b,0x21,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2e217be0 <unknown>

f1cvtl v31.8h, v31.8b
// CHECK-INST: f1cvtl v31.8h, v31.8b
// CHECK-ENCODING: [0xff,0x7b,0x21,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2e217bff <unknown>

///
/// F1CVTL2 instructions.
///
f1cvtl2 v0.8h, v0.16b
// CHECK-INST: f1cvtl2 v0.8h, v0.16b
// CHECK-ENCODING: [0x00,0x78,0x21,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6e217800 <unknown>

f1cvtl2 v0.8h, v31.16b
// CHECK-INST: f1cvtl2 v0.8h, v31.16b
// CHECK-ENCODING: [0xe0,0x7b,0x21,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6e217be0 <unknown>

f1cvtl2 v31.8h, v31.16b
// CHECK-INST: f1cvtl2 v31.8h, v31.16b
// CHECK-ENCODING: [0xff,0x7b,0x21,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6e217bff <unknown>

///
/// F2CVTL instructions.
///
f2cvtl v0.8h, v0.8b
// CHECK-INST: f2cvtl v0.8h, v0.8b
// CHECK-ENCODING: [0x00,0x78,0x61,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2e617800 <unknown>

f2cvtl v0.8h, v31.8b
// CHECK-INST: f2cvtl v0.8h, v31.8b
// CHECK-ENCODING: [0xe0,0x7b,0x61,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2e617be0 <unknown>

f2cvtl v31.8h, v31.8b
// CHECK-INST: f2cvtl v31.8h, v31.8b
// CHECK-ENCODING: [0xff,0x7b,0x61,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2e617bff <unknown>

///
/// F2CVTL2 instructions.
///
f2cvtl2 v0.8h, v0.16b
// CHECK-INST: f2cvtl2 v0.8h, v0.16b
// CHECK-ENCODING: [0x00,0x78,0x61,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6e617800 <unknown>

f2cvtl2 v0.8h, v31.16b
// CHECK-INST: f2cvtl2 v0.8h, v31.16b
// CHECK-ENCODING: [0xe0,0x7b,0x61,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6e617be0 <unknown>

f2cvtl2 v31.8h, v31.16b
// CHECK-INST: f2cvtl2 v31.8h, v31.16b
// CHECK-ENCODING: [0xff,0x7b,0x61,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6e617bff <unknown>

///
/// FCVTN instructions.
///
// FP16 TO FP8
fcvtn  v31.8b, v31.4h, v31.4h
// CHECK-INST: fcvtn  v31.8b, v31.4h, v31.4h
// CHECK-ENCODING: [0xff,0xf7,0x5f,0x0e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 0e5ff7ff <unknown>

fcvtn  v31.8b, v0.4h, v0.4h
// CHECK-INST: fcvtn  v31.8b, v0.4h, v0.4h
// CHECK-ENCODING: [0x1f,0xf4,0x40,0x0e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 0e40f41f <unknown>

fcvtn  v0.8b, v0.4h, v0.4h
// CHECK-INST: fcvtn  v0.8b, v0.4h, v0.4h
// CHECK-ENCODING: [0x00,0xf4,0x40,0x0e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 0e40f400 <unknown>

fcvtn  v0.16b, v0.8h, v0.8h
// CHECK-INST: fcvtn  v0.16b, v0.8h, v0.8h
// CHECK-ENCODING: [0x00,0xf4,0x40,0x4e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 4e40f400 <unknown>

fcvtn  v31.16b, v0.8h, v0.8h
// CHECK-INST: fcvtn  v31.16b, v0.8h, v0.8h
// CHECK-ENCODING: [0x1f,0xf4,0x40,0x4e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 4e40f41f <unknown>

fcvtn  v31.16b, v31.8h, v31.8h
// CHECK-INST: fcvtn  v31.16b, v31.8h, v31.8h
// CHECK-ENCODING: [0xff,0xf7,0x5f,0x4e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 4e5ff7ff <unknown>

// FP32 TO FP8
fcvtn  v0.8b, v0.4s, v0.4s
// CHECK-INST: fcvtn  v0.8b, v0.4s, v0.4s
// CHECK-ENCODING: [0x00,0xf4,0x00,0x0e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 0e00f400 <unknown>

fcvtn  v0.8b, v31.4s, v31.4s
// CHECK-INST: fcvtn  v0.8b, v31.4s, v31.4s
// CHECK-ENCODING: [0xe0,0xf7,0x1f,0x0e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 0e1ff7e0 <unknown>

fcvtn  v31.8b, v31.4s, v31.4s
// CHECK-INST: fcvtn  v31.8b, v31.4s, v31.4s
// CHECK-ENCODING: [0xff,0xf7,0x1f,0x0e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 0e1ff7ff <unknown>

///
/// FCVTN2 instructions.
///

fcvtn2  v0.16b, v0.4s, v0.4s
// CHECK-INST: fcvtn2  v0.16b, v0.4s, v0.4s
// CHECK-ENCODING: [0x00,0xf4,0x00,0x4e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 4e00f400 <unknown>

fcvtn2  v0.16b, v0.4s, v31.4s
// CHECK-INST: fcvtn2  v0.16b, v0.4s, v31.4s
// CHECK-ENCODING: [0x00,0xf4,0x1f,0x4e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 4e1ff400 <unknown>

fcvtn2  v31.16b, v31.4s, v31.4s
// CHECK-INST: fcvtn2  v31.16b, v31.4s, v31.4s
// CHECK-ENCODING: [0xff,0xf7,0x1f,0x4e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 4e1ff7ff <unknown>

///
/// FSCALE instructions.
///
fscale  v0.4h, v0.4h, v0.4h
// CHECK-INST: fscale  v0.4h, v0.4h, v0.4h
// CHECK-ENCODING: [0x00,0x3c,0xc0,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ec03c00 <unknown>

fscale  v0.4h, v31.4h, v31.4h
// CHECK-INST: fscale  v0.4h, v31.4h, v31.4h
// CHECK-ENCODING: [0xe0,0x3f,0xdf,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2edf3fe0 <unknown>

fscale  v31.4h, v31.4h, v31.4h
// CHECK-INST: fscale  v31.4h, v31.4h, v31.4h
// CHECK-ENCODING: [0xff,0x3f,0xdf,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2edf3fff <unknown>

fscale  v0.8h, v0.8h, v0.8h
// CHECK-INST: fscale  v0.8h, v0.8h, v0.8h
// CHECK-ENCODING: [0x00,0x3c,0xc0,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ec03c00 <unknown>

fscale  v31.8h, v0.8h, v0.8h
// CHECK-INST: fscale  v31.8h, v0.8h, v0.8h
// CHECK-ENCODING: [0x1f,0x3c,0xc0,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ec03c1f <unknown>

fscale  v31.8h, v31.8h, v31.8h
// CHECK-INST: fscale  v31.8h, v31.8h, v31.8h
// CHECK-ENCODING: [0xff,0x3f,0xdf,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6edf3fff <unknown>

fscale  v0.2s, v0.2s, v0.2s
// CHECK-INST: fscale  v0.2s, v0.2s, v0.2s
// CHECK-ENCODING: [0x00,0xfc,0xa0,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ea0fc00 <unknown>

fscale  v0.2s, v0.2s, v31.2s
// CHECK-INST: fscale  v0.2s, v0.2s, v31.2s
// CHECK-ENCODING: [0x00,0xfc,0xbf,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ebffc00 <unknown>

fscale  v31.2s, v31.2s, v31.2s
// CHECK-INST: fscale  v31.2s, v31.2s, v31.2s
// CHECK-ENCODING: [0xff,0xff,0xbf,0x2e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 2ebfffff <unknown>

fscale  v0.4s, v0.4s, v0.4s
// CHECK-INST: fscale  v0.4s, v0.4s, v0.4s
// CHECK-ENCODING: [0x00,0xfc,0xa0,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ea0fc00 <unknown>

fscale  v0.4s, v31.4s, v0.4s
// CHECK-INST: fscale  v0.4s, v31.4s, v0.4s
// CHECK-ENCODING: [0xe0,0xff,0xa0,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ea0ffe0 <unknown>

fscale  v31.4s, v31.4s, v31.4s
// CHECK-INST: fscale  v31.4s, v31.4s, v31.4s
// CHECK-ENCODING: [0xff,0xff,0xbf,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ebfffff <unknown>

fscale  v0.2d, v0.2d, v0.2d
// CHECK-INST: fscale  v0.2d, v0.2d, v0.2d
// CHECK-ENCODING: [0x00,0xfc,0xe0,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ee0fc00 <unknown>

fscale  v0.2d, v31.2d, v0.2d
// CHECK-INST: fscale  v0.2d, v31.2d, v0.2d
// CHECK-ENCODING: [0xe0,0xff,0xe0,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6ee0ffe0 <unknown>

fscale  v31.2d, v31.2d, v31.2d
// CHECK-INST: fscale  v31.2d, v31.2d, v31.2d
// CHECK-ENCODING: [0xff,0xff,0xff,0x6e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 6effffff
