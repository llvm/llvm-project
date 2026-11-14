// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fpv2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fpv2 < %s \
// RUN:        | llvm-objdump -d --mattr=+fpv2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fpv2 < %s \
// RUN:        | llvm-objdump -d --mattr=-fpv2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fpv2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+fpv2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// FMAXINM

fmaxinm h0, h0, h0
// CHECK-INST: fmaxinm h0, h0, h0
// CHECK-ENCODING: encoding: [0x00,0xa8,0xe0,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1ee0a800 <unknown>

fmaxinm h31, h31, h31
// CHECK-INST: fmaxinm h31, h31, h31
// CHECK-ENCODING: encoding: [0xff,0xab,0xff,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1effabff <unknown>

fmaxinm s0, s0, s0
// CHECK-INST: fmaxinm s0, s0, s0
// CHECK-ENCODING: encoding: [0x00,0xa8,0x20,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e20a800 <unknown>

fmaxinm s31, s31, s31
// CHECK-INST: fmaxinm s31, s31, s31
// CHECK-ENCODING: encoding: [0xff,0xab,0x3f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e3fabff <unknown>

fmaxinm d0, d0, d0
// CHECK-INST: fmaxinm d0, d0, d0
// CHECK-ENCODING: encoding: [0x00,0xa8,0x60,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e60a800 <unknown>

fmaxinm d31, d31, d31
// CHECK-INST: fmaxinm d31, d31, d31
// CHECK-ENCODING: encoding: [0xff,0xab,0x7f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e7fabff <unknown>


// FMININM

fmininm h0, h0, h0
// CHECK-INST: fmininm h0, h0, h0
// CHECK-ENCODING: encoding: [0x00,0xb8,0xe0,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1ee0b800 <unknown>

fmininm h31, h31, h31
// CHECK-INST: fmininm h31, h31, h31
// CHECK-ENCODING: encoding: [0xff,0xbb,0xff,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1effbbff <unknown>

fmininm s0, s0, s0
// CHECK-INST: fmininm s0, s0, s0
// CHECK-ENCODING: encoding: [0x00,0xb8,0x20,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e20b800 <unknown>

fmininm s31, s31, s31
// CHECK-INST: fmininm s31, s31, s31
// CHECK-ENCODING: encoding: [0xff,0xbb,0x3f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e3fbbff <unknown>

fmininm d0, d0, d0
// CHECK-INST: fmininm d0, d0, d0
// CHECK-ENCODING: encoding: [0x00,0xb8,0x60,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e60b800 <unknown>

fmininm d31, d31, d31
// CHECK-INST: fmininm d31, d31, d31
// CHECK-ENCODING: encoding: [0xff,0xbb,0x7f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e7fbbff <unknown>


// FMAXMG

fmaxmg h0, h0, h0
// CHECK-INST: fmaxmg h0, h0, h0
// CHECK-ENCODING: encoding: [0x00,0xc8,0xe0,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1ee0c800 <unknown>

fmaxmg h31, h31, h31
// CHECK-INST: fmaxmg h31, h31, h31
// CHECK-ENCODING: encoding: [0xff,0xcb,0xff,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1effcbff <unknown>

fmaxmg s0, s0, s0
// CHECK-INST: fmaxmg s0, s0, s0
// CHECK-ENCODING: encoding: [0x00,0xc8,0x20,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e20c800 <unknown>

fmaxmg s31, s31, s31
// CHECK-INST: fmaxmg s31, s31, s31
// CHECK-ENCODING: encoding: [0xff,0xcb,0x3f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e3fcbff <unknown>

fmaxmg d0, d0, d0
// CHECK-INST: fmaxmg d0, d0, d0
// CHECK-ENCODING: encoding: [0x00,0xc8,0x60,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e60c800 <unknown>

fmaxmg d31, d31, d31
// CHECK-INST: fmaxmg d31, d31, d31
// CHECK-ENCODING: encoding: [0xff,0xcb,0x7f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e7fcbff <unknown>


// FMINMG

fminmg h0, h0, h0
// CHECK-INST: fminmg h0, h0, h0
// CHECK-ENCODING: encoding: [0x00,0xd8,0xe0,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1ee0d800 <unknown>

fminmg h31, h31, h31
// CHECK-INST: fminmg h31, h31, h31
// CHECK-ENCODING: encoding: [0xff,0xdb,0xff,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1effdbff <unknown>

fminmg s0, s0, s0
// CHECK-INST: fminmg s0, s0, s0
// CHECK-ENCODING: encoding: [0x00,0xd8,0x20,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e20d800 <unknown>

fminmg s31, s31, s31
// CHECK-INST: fminmg s31, s31, s31
// CHECK-ENCODING: encoding: [0xff,0xdb,0x3f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e3fdbff <unknown>

fminmg d0, d0, d0
// CHECK-INST: fminmg d0, d0, d0
// CHECK-ENCODING: encoding: [0x00,0xd8,0x60,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e60d800 <unknown>

fminmg d31, d31, d31
// CHECK-INST: fminmg d31, d31, d31
// CHECK-ENCODING: encoding: [0xff,0xdb,0x7f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e7fdbff <unknown>


// FMAXMGINM

fmaxmginm h0, h0, h0
// CHECK-INST: fmaxmginm h0, h0, h0
// CHECK-ENCODING: encoding: [0x00,0xe8,0xe0,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1ee0e800 <unknown>

fmaxmginm h31, h31, h31
// CHECK-INST: fmaxmginm h31, h31, h31
// CHECK-ENCODING: encoding: [0xff,0xeb,0xff,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1effebff <unknown>

fmaxmginm s0, s0, s0
// CHECK-INST: fmaxmginm s0, s0, s0
// CHECK-ENCODING: encoding: [0x00,0xe8,0x20,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e20e800 <unknown>

fmaxmginm s31, s31, s31
// CHECK-INST: fmaxmginm s31, s31, s31
// CHECK-ENCODING: encoding: [0xff,0xeb,0x3f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e3febff <unknown>

fmaxmginm d0, d0, d0
// CHECK-INST: fmaxmginm d0, d0, d0
// CHECK-ENCODING: encoding: [0x00,0xe8,0x60,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e60e800 <unknown>

fmaxmginm d31, d31, d31
// CHECK-INST: fmaxmginm d31, d31, d31
// CHECK-ENCODING: encoding: [0xff,0xeb,0x7f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e7febff <unknown>


// FMINMGINM

fminmginm h0, h0, h0
// CHECK-INST: fminmginm h0, h0, h0
// CHECK-ENCODING: encoding: [0x00,0xf8,0xe0,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1ee0f800 <unknown>

fminmginm h31, h31, h31
// CHECK-INST: fminmginm h31, h31, h31
// CHECK-ENCODING: encoding: [0xff,0xfb,0xff,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1efffbff <unknown>

fminmginm s0, s0, s0
// CHECK-INST: fminmginm s0, s0, s0
// CHECK-ENCODING: encoding: [0x00,0xf8,0x20,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e20f800 <unknown>

fminmginm s31, s31, s31
// CHECK-INST: fminmginm s31, s31, s31
// CHECK-ENCODING: encoding: [0xff,0xfb,0x3f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e3ffbff <unknown>

fminmginm d0, d0, d0
// CHECK-INST: fminmginm d0, d0, d0
// CHECK-ENCODING: encoding: [0x00,0xf8,0x60,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e60f800 <unknown>

fminmginm d31, d31, d31
// CHECK-INST: fminmginm d31, d31, d31
// CHECK-ENCODING: encoding: [0xff,0xfb,0x7f,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e7ffbff <unknown>
