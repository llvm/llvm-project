// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

dup     z0.b, w0
// CHECK-INST: mov     z0.b, w0
// CHECK-ENCODING: [0x00,0x38,0x20,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05203800 <unknown>

dup     z0.h, w0
// CHECK-INST: mov     z0.h, w0
// CHECK-ENCODING: [0x00,0x38,0x60,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05603800 <unknown>

dup     z0.s, w0
// CHECK-INST: mov     z0.s, w0
// CHECK-ENCODING: [0x00,0x38,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a03800 <unknown>

dup     z0.d, x0
// CHECK-INST: mov     z0.d, x0
// CHECK-ENCODING: [0x00,0x38,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e03800 <unknown>

dup     z31.h, wsp
// CHECK-INST: mov     z31.h, wsp
// CHECK-ENCODING: [0xff,0x3b,0x60,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05603bff <unknown>

dup     z31.s, wsp
// CHECK-INST: mov     z31.s, wsp
// CHECK-ENCODING: [0xff,0x3b,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a03bff <unknown>

dup     z31.d, sp
// CHECK-INST: mov     z31.d, sp
// CHECK-ENCODING: [0xff,0x3b,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e03bff <unknown>

dup     z31.b, wsp
// CHECK-INST: mov     z31.b, wsp
// CHECK-ENCODING: [0xff,0x3b,0x20,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05203bff <unknown>

dup     z5.b, #-128
// CHECK-INST: mov     z5.b, #-128
// CHECK-ENCODING: [0x05,0xd0,0x38,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2538d005 <unknown>

dup     z5.b, #127
// CHECK-INST: mov     z5.b, #127
// CHECK-ENCODING: [0xe5,0xcf,0x38,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2538cfe5 <unknown>

dup     z5.b, #255
// CHECK-INST: mov     z5.b, #-1
// CHECK-ENCODING: [0xe5,0xdf,0x38,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2538dfe5 <unknown>

dup     z21.h, #-128
// CHECK-INST: mov     z21.h, #-128
// CHECK-ENCODING: [0x15,0xd0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2578d015 <unknown>

dup     z21.h, #-128, lsl #8
// CHECK-INST: mov     z21.h, #-32768
// CHECK-ENCODING: [0x15,0xf0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2578f015 <unknown>

dup     z21.h, #-32768
// CHECK-INST: mov     z21.h, #-32768
// CHECK-ENCODING: [0x15,0xf0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2578f015 <unknown>

dup     z21.h, #127
// CHECK-INST: mov     z21.h, #127
// CHECK-ENCODING: [0xf5,0xcf,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2578cff5 <unknown>

dup     z21.h, #127, lsl #8
// CHECK-INST: mov     z21.h, #32512
// CHECK-ENCODING: [0xf5,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2578eff5 <unknown>

dup     z21.h, #32512
// CHECK-INST: mov     z21.h, #32512
// CHECK-ENCODING: [0xf5,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2578eff5 <unknown>

dup     z21.s, #-128
// CHECK-INST: mov     z21.s, #-128
// CHECK-ENCODING: [0x15,0xd0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25b8d015 <unknown>

dup     z21.s, #-128, lsl #8
// CHECK-INST: mov     z21.s, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25b8f015 <unknown>

dup     z21.s, #-32768
// CHECK-INST: mov     z21.s, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25b8f015 <unknown>

dup     z21.s, #127
// CHECK-INST: mov     z21.s, #127
// CHECK-ENCODING: [0xf5,0xcf,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25b8cff5 <unknown>

dup     z21.s, #127, lsl #8
// CHECK-INST: mov     z21.s, #32512
// CHECK-ENCODING: [0xf5,0xef,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25b8eff5 <unknown>

dup     z21.s, #32512
// CHECK-INST: mov     z21.s, #32512
// CHECK-ENCODING: [0xf5,0xef,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25b8eff5 <unknown>

dup     z21.d, #-128
// CHECK-INST: mov     z21.d, #-128
// CHECK-ENCODING: [0x15,0xd0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f8d015 <unknown>

dup     z21.d, #-128, lsl #8
// CHECK-INST: mov     z21.d, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f8f015 <unknown>

dup     z21.d, #-32768
// CHECK-INST: mov     z21.d, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f8f015 <unknown>

dup     z21.d, #127
// CHECK-INST: mov     z21.d, #127
// CHECK-ENCODING: [0xf5,0xcf,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f8cff5 <unknown>

dup     z21.d, #127, lsl #8
// CHECK-INST: mov     z21.d, #32512
// CHECK-ENCODING: [0xf5,0xef,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f8eff5 <unknown>

dup     z21.d, #32512
// CHECK-INST: mov     z21.d, #32512
// CHECK-ENCODING: [0xf5,0xef,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f8eff5 <unknown>

dup     z0.b, z0.b[0]
// CHECK-INST: mov     z0.b, b0
// CHECK-ENCODING: [0x00,0x20,0x21,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05212000 <unknown>

dup     z0.h, z0.h[0]
// CHECK-INST: mov     z0.h, h0
// CHECK-ENCODING: [0x00,0x20,0x22,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05222000 <unknown>

dup     z0.s, z0.s[0]
// CHECK-INST: mov     z0.s, s0
// CHECK-ENCODING: [0x00,0x20,0x24,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05242000 <unknown>

dup     z0.d, z0.d[0]
// CHECK-INST: mov     z0.d, d0
// CHECK-ENCODING: [0x00,0x20,0x28,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05282000 <unknown>

dup     z0.q, z0.q[0]
// CHECK-INST: mov     z0.q, q0
// CHECK-ENCODING: [0x00,0x20,0x30,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05302000 <unknown>

dup     z31.b, z31.b[63]
// CHECK-INST: mov     z31.b, z31.b[63]
// CHECK-ENCODING: [0xff,0x23,0xff,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05ff23ff <unknown>

dup     z31.h, z31.h[31]
// CHECK-INST: mov     z31.h, z31.h[31]
// CHECK-ENCODING: [0xff,0x23,0xfe,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05fe23ff <unknown>

dup     z31.s, z31.s[15]
// CHECK-INST: mov     z31.s, z31.s[15]
// CHECK-ENCODING: [0xff,0x23,0xfc,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05fc23ff <unknown>

dup     z31.d, z31.d[7]
// CHECK-INST: mov     z31.d, z31.d[7]
// CHECK-ENCODING: [0xff,0x23,0xf8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05f823ff <unknown>

dup     z5.q, z17.q[3]
// CHECK-INST: mov     z5.q, z17.q[3]
// CHECK-ENCODING: [0x25,0x22,0xf0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05f02225 <unknown>

// --------------------------------------------------------------------------//
// Tests where the negative immediate is in bounds when interpreted
// as the element type.

dup     z0.b, #-129
// CHECK-INST: mov     z0.b, #127
// CHECK-ENCODING: [0xe0,0xcf,0x38,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2538cfe0 <unknown>

dup     z0.h, #-33024
// CHECK-INST: mov     z0.h, #32512
// CHECK-ENCODING: [0xe0,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2578efe0 <unknown>

dup     z0.h, #-129, lsl #8
// CHECK-INST: mov     z0.h, #32512
// CHECK-ENCODING: [0xe0,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2578efe0 <unknown>
