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
// F1CVT instructions
//
f1cvt   z0.h, z0.b  // 01100101-00001000-00110000-00000000
// CHECK-INST: f1cvt   z0.h, z0.b
// CHECK-ENCODING: [0x00,0x30,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083000 <unknown>

f1cvt   z0.h, z31.b  // 01100101-00001000-00110011-11100000
// CHECK-INST: f1cvt   z0.h, z31.b
// CHECK-ENCODING: [0xe0,0x33,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650833e0 <unknown>

f1cvt   z31.h, z0.b  // 01100101-00001000-00110000-00011111
// CHECK-INST: f1cvt   z31.h, z0.b
// CHECK-ENCODING: [0x1f,0x30,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 6508301f <unknown>

f1cvt   z31.h, z31.b  // 01100101-00001000-00110011-11111111
// CHECK-INST: f1cvt   z31.h, z31.b
// CHECK-ENCODING: [0xff,0x33,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650833ff <unknown>

//
// F2CVT instructions
//
f2cvt   z0.h, z0.b  // 01100101-00001000-00110100-00000000
// CHECK-INST: f2cvt   z0.h, z0.b
// CHECK-ENCODING: [0x00,0x34,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083400 <unknown>

f2cvt   z0.h, z31.b  // 01100101-00001000-00110111-11100000
// CHECK-INST: f2cvt   z0.h, z31.b
// CHECK-ENCODING: [0xe0,0x37,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650837e0 <unknown>

f2cvt   z31.h, z0.b  // 01100101-00001000-00110100-00011111
// CHECK-INST: f2cvt   z31.h, z0.b
// CHECK-ENCODING: [0x1f,0x34,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 6508341f <unknown>

f2cvt   z31.h, z31.b  // 01100101-00001000-00110111-11111111
// CHECK-INST: f2cvt   z31.h, z31.b
// CHECK-ENCODING: [0xff,0x37,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650837ff <unknown>


//
// BF1CVT instructions
//
bf1cvt  z0.h, z0.b  // 01100101-00001000-00111000-00000000
// CHECK-INST: bf1cvt  z0.h, z0.b
// CHECK-ENCODING: [0x00,0x38,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083800 <unknown>

bf1cvt  z0.h, z31.b  // 01100101-00001000-00111011-11100000
// CHECK-INST: bf1cvt  z0.h, z31.b
// CHECK-ENCODING: [0xe0,0x3b,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083be0 <unknown>

bf1cvt  z31.h, z0.b  // 01100101-00001000-00111000-00011111
// CHECK-INST: bf1cvt  z31.h, z0.b
// CHECK-ENCODING: [0x1f,0x38,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 6508381f <unknown>

bf1cvt  z31.h, z31.b  // 01100101-00001000-00111011-11111111
// CHECK-INST: bf1cvt  z31.h, z31.b
// CHECK-ENCODING: [0xff,0x3b,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083bff <unknown>


//
// BF2CVT instructions
//
bf2cvt  z0.h, z0.b  // 01100101-00001000-00111100-00000000
// CHECK-INST: bf2cvt  z0.h, z0.b
// CHECK-ENCODING: [0x00,0x3c,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083c00 <unknown>

bf2cvt  z0.h, z31.b  // 01100101-00001000-00111111-11100000
// CHECK-INST: bf2cvt  z0.h, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083fe0 <unknown>

bf2cvt  z31.h, z0.b  // 01100101-00001000-00111100-00011111
// CHECK-INST: bf2cvt  z31.h, z0.b
// CHECK-ENCODING: [0x1f,0x3c,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083c1f <unknown>

bf2cvt  z31.h, z31.b  // 01100101-00001000-00111111-11111111
// CHECK-INST: bf2cvt  z31.h, z31.b
// CHECK-ENCODING: [0xff,0x3f,0x08,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65083fff <unknown>


//
// F1CVTLT instructions
//
f1cvtlt z0.h, z0.b  // 01100101-00001001-00110000-00000000
// CHECK-INST: f1cvtlt z0.h, z0.b
// CHECK-ENCODING: [0x00,0x30,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093000 <unknown>

f1cvtlt z0.h, z31.b  // 01100101-00001001-00110011-11100000
// CHECK-INST: f1cvtlt z0.h, z31.b
// CHECK-ENCODING: [0xe0,0x33,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650933e0 <unknown>

f1cvtlt z31.h, z0.b  // 01100101-00001001-00110000-00011111
// CHECK-INST: f1cvtlt z31.h, z0.b
// CHECK-ENCODING: [0x1f,0x30,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 6509301f <unknown>

f1cvtlt z31.h, z31.b  // 01100101-00001001-00110011-11111111
// CHECK-INST: f1cvtlt z31.h, z31.b
// CHECK-ENCODING: [0xff,0x33,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650933ff <unknown>


//
// F2CVTLT instructions
//
f2cvtlt z0.h, z0.b  // 01100101-00001001-00110100-00000000
// CHECK-INST: f2cvtlt z0.h, z0.b
// CHECK-ENCODING: [0x00,0x34,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093400 <unknown>

f2cvtlt z0.h, z31.b  // 01100101-00001001-00110111-11100000
// CHECK-INST: f2cvtlt z0.h, z31.b
// CHECK-ENCODING: [0xe0,0x37,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650937e0 <unknown>

f2cvtlt z31.h, z0.b  // 01100101-00001001-00110100-00011111
// CHECK-INST: f2cvtlt z31.h, z0.b
// CHECK-ENCODING: [0x1f,0x34,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 6509341f <unknown>

f2cvtlt z31.h, z31.b  // 01100101-00001001-00110111-11111111
// CHECK-INST: f2cvtlt z31.h, z31.b
// CHECK-ENCODING: [0xff,0x37,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 650937ff <unknown>


//
// BF1CVTLT instructions
//
bf1cvtlt z0.h, z0.b  // 01100101-00001001-00111000-00000000
// CHECK-INST: bf1cvtlt z0.h, z0.b
// CHECK-ENCODING: [0x00,0x38,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093800 <unknown>

bf1cvtlt z0.h, z31.b  // 01100101-00001001-00111011-11100000
// CHECK-INST: bf1cvtlt z0.h, z31.b
// CHECK-ENCODING: [0xe0,0x3b,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093be0 <unknown>

bf1cvtlt z31.h, z0.b  // 01100101-00001001-00111000-00011111
// CHECK-INST: bf1cvtlt z31.h, z0.b
// CHECK-ENCODING: [0x1f,0x38,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 6509381f <unknown>

bf1cvtlt z31.h, z31.b  // 01100101-00001001-00111011-11111111
// CHECK-INST: bf1cvtlt z31.h, z31.b
// CHECK-ENCODING: [0xff,0x3b,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093bff <unknown>


//
// BF2CVTLT instructions
//
bf2cvtlt z0.h, z0.b  // 01100101-00001001-00111100-00000000
// CHECK-INST: bf2cvtlt z0.h, z0.b
// CHECK-ENCODING: [0x00,0x3c,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093c00 <unknown>

bf2cvtlt z0.h, z31.b  // 01100101-00001001-00111111-11100000
// CHECK-INST: bf2cvtlt z0.h, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093fe0 <unknown>

bf2cvtlt z31.h, z0.b  // 01100101-00001001-00111100-00011111
// CHECK-INST: bf2cvtlt z31.h, z0.b
// CHECK-ENCODING: [0x1f,0x3c,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093c1f <unknown>

bf2cvtlt z31.h, z31.b  // 01100101-00001001-00111111-11111111
// CHECK-INST: bf2cvtlt z31.h, z31.b
// CHECK-ENCODING: [0xff,0x3f,0x09,0x65]
// CHECK-ERROR: instruction requires: fp8 sve2
// CHECK-UNKNOWN: 65093fff <unknown>
