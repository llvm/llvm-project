// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

dup     z0.b, w0
// CHECK-INST: mov     z0.b, w0
// CHECK-ENCODING: [0x00,0x38,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 38 20 05 <unknown>

dup     z0.h, w0
// CHECK-INST: mov     z0.h, w0
// CHECK-ENCODING: [0x00,0x38,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 38 60 05 <unknown>

dup     z0.s, w0
// CHECK-INST: mov     z0.s, w0
// CHECK-ENCODING: [0x00,0x38,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 38 a0 05 <unknown>

dup     z0.d, x0
// CHECK-INST: mov     z0.d, x0
// CHECK-ENCODING: [0x00,0x38,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 38 e0 05 <unknown>

dup     z31.h, wsp
// CHECK-INST: mov     z31.h, wsp
// CHECK-ENCODING: [0xff,0x3b,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b 60 05 <unknown>

dup     z31.s, wsp
// CHECK-INST: mov     z31.s, wsp
// CHECK-ENCODING: [0xff,0x3b,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b a0 05 <unknown>

dup     z31.d, sp
// CHECK-INST: mov     z31.d, sp
// CHECK-ENCODING: [0xff,0x3b,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b e0 05 <unknown>

dup     z31.b, wsp
// CHECK-INST: mov     z31.b, wsp
// CHECK-ENCODING: [0xff,0x3b,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b 20 05 <unknown>

dup     z5.b, #-128
// CHECK-INST: mov     z5.b, #-128
// CHECK-ENCODING: [0x05,0xd0,0x38,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 d0 38 25 <unknown>

dup     z5.b, #127
// CHECK-INST: mov     z5.b, #127
// CHECK-ENCODING: [0xe5,0xcf,0x38,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 cf 38 25 <unknown>

dup     z5.b, #255
// CHECK-INST: mov     z5.b, #-1
// CHECK-ENCODING: [0xe5,0xdf,0x38,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 df 38 25 <unknown>

dup     z21.h, #-128
// CHECK-INST: mov     z21.h, #-128
// CHECK-ENCODING: [0x15,0xd0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 d0 78 25 <unknown>

dup     z21.h, #-128, lsl #8
// CHECK-INST: mov     z21.h, #-32768
// CHECK-ENCODING: [0x15,0xf0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 78 25 <unknown>

dup     z21.h, #-32768
// CHECK-INST: mov     z21.h, #-32768
// CHECK-ENCODING: [0x15,0xf0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 78 25 <unknown>

dup     z21.h, #127
// CHECK-INST: mov     z21.h, #127
// CHECK-ENCODING: [0xf5,0xcf,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 cf 78 25 <unknown>

dup     z21.h, #127, lsl #8
// CHECK-INST: mov     z21.h, #32512
// CHECK-ENCODING: [0xf5,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef 78 25 <unknown>

dup     z21.h, #32512
// CHECK-INST: mov     z21.h, #32512
// CHECK-ENCODING: [0xf5,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef 78 25 <unknown>

dup     z21.s, #-128
// CHECK-INST: mov     z21.s, #-128
// CHECK-ENCODING: [0x15,0xd0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 d0 b8 25 <unknown>

dup     z21.s, #-128, lsl #8
// CHECK-INST: mov     z21.s, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 b8 25 <unknown>

dup     z21.s, #-32768
// CHECK-INST: mov     z21.s, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 b8 25 <unknown>

dup     z21.s, #127
// CHECK-INST: mov     z21.s, #127
// CHECK-ENCODING: [0xf5,0xcf,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 cf b8 25 <unknown>

dup     z21.s, #127, lsl #8
// CHECK-INST: mov     z21.s, #32512
// CHECK-ENCODING: [0xf5,0xef,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef b8 25 <unknown>

dup     z21.s, #32512
// CHECK-INST: mov     z21.s, #32512
// CHECK-ENCODING: [0xf5,0xef,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef b8 25 <unknown>

dup     z21.d, #-128
// CHECK-INST: mov     z21.d, #-128
// CHECK-ENCODING: [0x15,0xd0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 d0 f8 25 <unknown>

dup     z21.d, #-128, lsl #8
// CHECK-INST: mov     z21.d, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 f8 25 <unknown>

dup     z21.d, #-32768
// CHECK-INST: mov     z21.d, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 f8 25 <unknown>

dup     z21.d, #127
// CHECK-INST: mov     z21.d, #127
// CHECK-ENCODING: [0xf5,0xcf,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 cf f8 25 <unknown>

dup     z21.d, #127, lsl #8
// CHECK-INST: mov     z21.d, #32512
// CHECK-ENCODING: [0xf5,0xef,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef f8 25 <unknown>

dup     z21.d, #32512
// CHECK-INST: mov     z21.d, #32512
// CHECK-ENCODING: [0xf5,0xef,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef f8 25 <unknown>
