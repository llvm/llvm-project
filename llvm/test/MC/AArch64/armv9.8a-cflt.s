// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cflt < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cflt < %s \
// RUN:        | llvm-objdump -d --mattr=+cflt --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cflt < %s \
// RUN:        | llvm-objdump -d --mattr=-cflt --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cflt < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+cflt -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// Conditional Fault instructions (FEAT_CFLT)
//------------------------------------------------------------------------------

flt.eq #0
// CHECK-INST: flt.eq #0
// CHECK-ENCODING: [0x10,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d4200010
// CHECK-ERROR: error: instruction requires: cflt

flt.ne #1
// CHECK-INST: flt.ne #1
// CHECK-ENCODING: [0x31,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d4200031
// CHECK-ERROR: error: instruction requires: cflt

flt.cs #2
// CHECK-INST: flt.hs #2
// CHECK-ENCODING: [0x52,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d4200052
// CHECK-ERROR: error: instruction requires: cflt

flt.hs #2
// CHECK-INST: flt.hs #2
// CHECK-ENCODING: [0x52,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d4200052
// CHECK-ERROR: error: instruction requires: cflt

flt.cc #3
// CHECK-INST: flt.lo #3
// CHECK-ENCODING: [0x73,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d4200073
// CHECK-ERROR: error: instruction requires: cflt

flt.lo #3
// CHECK-INST: flt.lo #3
// CHECK-ENCODING: [0x73,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d4200073
// CHECK-ERROR: error: instruction requires: cflt

flt.mi #4
// CHECK-INST: flt.mi #4
// CHECK-ENCODING: [0x94,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d4200094
// CHECK-ERROR: error: instruction requires: cflt

flt.pl #5
// CHECK-INST: flt.pl #5
// CHECK-ENCODING: [0xb5,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d42000b5
// CHECK-ERROR: error: instruction requires: cflt

flt.vs #6
// CHECK-INST: flt.vs #6
// CHECK-ENCODING: [0xd6,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d42000d6
// CHECK-ERROR: error: instruction requires: cflt

flt.vc #7
// CHECK-INST: flt.vc #7
// CHECK-ENCODING: [0xf7,0x00,0x20,0xd4]
// CHECK-UNKNOWN: d42000f7
// CHECK-ERROR: error: instruction requires: cflt

flt.hi #8
// CHECK-INST: flt.hi #8
// CHECK-ENCODING: [0x18,0x01,0x20,0xd4]
// CHECK-UNKNOWN: d4200118
// CHECK-ERROR: error: instruction requires: cflt

flt.ls #9
// CHECK-INST: flt.ls #9
// CHECK-ENCODING: [0x39,0x01,0x20,0xd4]
// CHECK-UNKNOWN: d4200139
// CHECK-ERROR: error: instruction requires: cflt

flt.ge #10
// CHECK-INST: flt.ge #10
// CHECK-ENCODING: [0x5a,0x01,0x20,0xd4]
// CHECK-UNKNOWN: d420015a
// CHECK-ERROR: error: instruction requires: cflt

flt.lt #11
// CHECK-INST: flt.lt #11
// CHECK-ENCODING: [0x7b,0x01,0x20,0xd4]
// CHECK-UNKNOWN: d420017b
// CHECK-ERROR: error: instruction requires: cflt

flt.gt #12
// CHECK-INST: flt.gt #12
// CHECK-ENCODING: [0x9c,0x01,0x20,0xd4]
// CHECK-UNKNOWN: d420019c
// CHECK-ERROR: error: instruction requires: cflt

flt.le #13
// CHECK-INST: flt.le #13
// CHECK-ENCODING: [0xbd,0x01,0x20,0xd4]
// CHECK-UNKNOWN: d42001bd
// CHECK-ERROR: error: instruction requires: cflt

flt.al #65535
// CHECK-INST: flt.al #65535
// CHECK-ENCODING: [0xfe,0xff,0x3f,0xd4]
// CHECK-UNKNOWN: d43ffffe
// CHECK-ERROR: error: instruction requires: cflt

flt.nv #15
// CHECK-INST: flt.nv #15
// CHECK-ENCODING: [0xff,0x01,0x20,0xd4]
// CHECK-UNKNOWN: d42001ff
// CHECK-ERROR: error: instruction requires: cflt

// CFLT<cc> (immediate).

cflteq #0, w0, #1
// CHECK-INST: cflteq #0, w0, #1
// CHECK-ENCODING: [0x20,0x00,0x80,0x76]
// CHECK-UNKNOWN: 76800020
// CHECK-ERROR: error: instruction requires: cflt

cfltne #3, x3, #-1
// CHECK-INST: cfltne #3, x3, #-1
// CHECK-ENCODING: [0xe3,0x0d,0xbf,0xf6]
// CHECK-UNKNOWN: f6bf0de3
// CHECK-ERROR: error: instruction requires: cflt

cfltgt #0, w4, #-256
// CHECK-INST: cfltgt #0, w4, #-256
// CHECK-ENCODING: [0x04,0x00,0x10,0x76]
// CHECK-UNKNOWN: 76100004
// CHECK-ERROR: error: instruction requires: cflt

cfltlt #3, x7, #0
// CHECK-INST: cfltlt #3, x7, #0
// CHECK-ENCODING: [0x07,0x0c,0x20,0xf6]
// CHECK-UNKNOWN: f6200c07
// CHECK-ERROR: error: instruction requires: cflt

cflthi #1, x9, #511
// CHECK-INST: cflthi #1, x9, #511
// CHECK-ENCODING: [0xe9,0x05,0x5f,0xf6]
// CHECK-UNKNOWN: f65f05e9
// CHECK-ERROR: error: instruction requires: cflt

cfltlo #2, w10, #1
// CHECK-INST: cfltlo #2, w10, #1
// CHECK-ENCODING: [0x2a,0x08,0x60,0x76]
// CHECK-UNKNOWN: 7660082a
// CHECK-ERROR: error: instruction requires: cflt

// CFLT<cc> (immediate) aliases.

cfltge #0, x0, #(2 - 1)
// CHECK-INST: cfltgt #0, x0, #0
// CHECK-ENCODING: [0x00,0x00,0x00,0xf6]
// CHECK-UNKNOWN: f6000000
// CHECK-ERROR: error: instruction requires: cflt

cfltge #1, x13, #256
// CHECK-INST: cfltgt #1, x13, #255
// CHECK-ENCODING: [0xed,0x05,0x0f,0xf6]
// CHECK-UNKNOWN: f60f05ed
// CHECK-ERROR: error: instruction requires: cflt

cflths #2, w14, #1
// CHECK-INST: cflthi #2, w14, #0
// CHECK-ENCODING: [0x0e,0x08,0x40,0x76]
// CHECK-UNKNOWN: 7640080e
// CHECK-ERROR: error: instruction requires: cflt

cflths #2, w14, #+1
// CHECK-INST: cflthi #2, w14, #0
// CHECK-ENCODING: [0x0e,0x08,0x40,0x76]
// CHECK-UNKNOWN: 7640080e
// CHECK-ERROR: error: instruction requires: cflt

cfltle #0, w16, #-257
// CHECK-INST: cfltlt #0, w16, #-256
// CHECK-ENCODING: [0x10,0x00,0x30,0x76]
// CHECK-UNKNOWN: 76300010
// CHECK-ERROR: error: instruction requires: cflt

cfltle #0, w16, #(-257)
// CHECK-INST: cfltlt #0, w16, #-256
// CHECK-ENCODING: [0x10,0x00,0x30,0x76]
// CHECK-UNKNOWN: 76300010
// CHECK-ERROR: error: instruction requires: cflt

cfltls #3, x19, #510
// CHECK-INST: cfltlo #3, x19, #511
// CHECK-ENCODING: [0xf3,0x0d,0x7f,0xf6]
// CHECK-UNKNOWN: f67f0df3
// CHECK-ERROR: error: instruction requires: cflt

// CFLT<cc> (register).

cflteq #0, w0, wsp
// CHECK-INST: cflteq #0, w0, wsp
// CHECK-ENCODING: [0x00,0x02,0xdf,0x76]
// CHECK-UNKNOWN: 76df0200
// CHECK-ERROR: error: instruction requires: cflt

cfltne #3, x3, x4
// CHECK-INST: cfltne #3, x3, x4
// CHECK-ENCODING: [0x03,0x0e,0xe4,0xf6]
// CHECK-UNKNOWN: f6e40e03
// CHECK-ERROR: error: instruction requires: cflt

cfltgt #0, w4, w5
// CHECK-INST: cfltgt #0, w4, w5
// CHECK-ENCODING: [0x04,0x02,0x05,0x76]
// CHECK-UNKNOWN: 76050204
// CHECK-ERROR: error: instruction requires: cflt

cfltge #3, x7, x8
// CHECK-INST: cfltge #3, x7, x8
// CHECK-ENCODING: [0x07,0x0e,0x28,0xf6]
// CHECK-UNKNOWN: f6280e07
// CHECK-ERROR: error: instruction requires: cflt

cflthi #0, w8, w9
// CHECK-INST: cflthi #0, w8, w9
// CHECK-ENCODING: [0x08,0x02,0x49,0x76]
// CHECK-UNKNOWN: 76490208
// CHECK-ERROR: error: instruction requires: cflt

cflths #3, x11, x12
// CHECK-INST: cflths #3, x11, x12
// CHECK-ENCODING: [0x0b,0x0e,0x6c,0xf6]
// CHECK-UNKNOWN: f66c0e0b
// CHECK-ERROR: error: instruction requires: cflt

// CFLT<cc> (register) aliases.

cfltle #0, w13, w14
// CHECK-INST: cfltge #0, w14, w13
// CHECK-ENCODING: [0x0e,0x02,0x2d,0x76]
// CHECK-UNKNOWN: 762d020e
// CHECK-ERROR: error: instruction requires: cflt

cfltlo #3, x16, x17
// CHECK-INST: cflthi #3, x17, x16
// CHECK-ENCODING: [0x11,0x0e,0x50,0xf6]
// CHECK-UNKNOWN: f6500e11
// CHECK-ERROR: error: instruction requires: cflt

cfltls #0, w17, w18
// CHECK-INST: cflths #0, w18, w17
// CHECK-ENCODING: [0x12,0x02,0x71,0x76]
// CHECK-UNKNOWN: 76710212
// CHECK-ERROR: error: instruction requires: cflt

cfltlt #3, x20, x21
// CHECK-INST: cfltgt #3, x21, x20
// CHECK-ENCODING: [0x15,0x0e,0x14,0xf6]
// CHECK-UNKNOWN: f6140e15
// CHECK-ERROR: error: instruction requires: cflt

// CFLTZ and CFLTNZ.

cfltz #0, w21
// CHECK-INST: cflteq #0, w21, #0
// CHECK-ENCODING: [0x15,0x00,0x80,0x76]
// CHECK-UNKNOWN: 76800015
// CHECK-ERROR: error: instruction requires: cflt

cfltnz #3, x24
// CHECK-INST: cfltne #3, x24, #0
// CHECK-ENCODING: [0x18,0x0c,0xa0,0xf6]
// CHECK-UNKNOWN: f6a00c18
// CHECK-ERROR: error: instruction requires: cflt

cfltnz #3, xzr
// CHECK-INST: cfltne #3, xzr, #0
// CHECK-ENCODING: [0x1f,0x0c,0xa0,0xf6]
// CHECK-UNKNOWN: f6a00c1f
// CHECK-ERROR: error: instruction requires: cflt

// TFLTZ and TFLTNZ.

tfltz #0, w0, #0
// CHECK-INST: tfltz #0, w0, #0
// CHECK-ENCODING: [0x40,0x02,0x00,0x76]
// CHECK-UNKNOWN: 76000240
// CHECK-ERROR: error: instruction requires: cflt

tfltz #0, x0, #0
// CHECK-INST: tfltz #0, w0, #0
// CHECK-ENCODING: [0x40,0x02,0x00,0x76]
// CHECK-UNKNOWN: 76000240
// CHECK-ERROR: error: instruction requires: cflt

tfltz #1, x1, #32
// CHECK-INST: tfltz #1, x1, #32
// CHECK-ENCODING: [0x41,0x06,0x00,0xf6]
// CHECK-UNKNOWN: f6000641
// CHECK-ERROR: error: instruction requires: cflt

tfltz #3, wzr, #31
// CHECK-INST: tfltz #3, wzr, #31
// CHECK-ENCODING: [0x5f,0x0e,0xf8,0x76]
// CHECK-UNKNOWN: 76f80e5f
// CHECK-ERROR: error: instruction requires: cflt

tfltnz #2, w3, #31
// CHECK-INST: tfltnz #2, w3, #31
// CHECK-ENCODING: [0x63,0x0a,0xf8,0x76]
// CHECK-UNKNOWN: 76f80a63
// CHECK-ERROR: error: instruction requires: cflt

tfltnz #3, xzr, #31
// CHECK-INST: tfltnz #3, wzr, #31
// CHECK-ENCODING: [0x7f,0x0e,0xf8,0x76]
// CHECK-UNKNOWN: 76f80e7f
// CHECK-ERROR: error: instruction requires: cflt

tfltnz #3, xzr, #32
// CHECK-INST: tfltnz #3, xzr, #32
// CHECK-ENCODING: [0x7f,0x0e,0x00,0xf6]
// CHECK-UNKNOWN: f6000e7f
// CHECK-ERROR: error: instruction requires: cflt

tfltnz #3, x4, #63
// CHECK-INST: tfltnz #3, x4, #63
// CHECK-ENCODING: [0x64,0x0e,0xf8,0xf6]
// CHECK-UNKNOWN: f6f80e64
// CHECK-ERROR: error: instruction requires: cflt
