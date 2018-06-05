// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

cpy     z5.b, p0/z, #-128
// CHECK-INST: mov     z5.b, p0/z, #-128
// CHECK-ENCODING: [0x05,0x10,0x10,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 10 10 05  <unknown>

cpy     z5.b, p0/z, #127
// CHECK-INST: mov     z5.b, p0/z, #127
// CHECK-ENCODING: [0xe5,0x0f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 0f 10 05  <unknown>

cpy     z5.b, p0/z, #255
// CHECK-INST: mov     z5.b, p0/z, #-1
// CHECK-ENCODING: [0xe5,0x1f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 1f 10 05  <unknown>

cpy     z21.h, p0/z, #-128
// CHECK-INST: mov     z21.h, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 10 50 05  <unknown>

cpy     z21.h, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.h, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 50 05  <unknown>

cpy     z21.h, p0/z, #-32768
// CHECK-INST: mov     z21.h, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 50 05  <unknown>

cpy     z21.h, p0/z, #127
// CHECK-INST: mov     z21.h, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 0f 50 05  <unknown>

cpy     z21.h, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.h, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f 50 05  <unknown>

cpy     z21.h, p0/z, #32512
// CHECK-INST: mov     z21.h, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f 50 05  <unknown>

cpy     z21.s, p0/z, #-128
// CHECK-INST: mov     z21.s, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 10 90 05  <unknown>

cpy     z21.s, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.s, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 90 05  <unknown>

cpy     z21.s, p0/z, #-32768
// CHECK-INST: mov     z21.s, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 90 05  <unknown>

cpy     z21.s, p0/z, #127
// CHECK-INST: mov     z21.s, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 0f 90 05  <unknown>

cpy     z21.s, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.s, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f 90 05  <unknown>

cpy     z21.s, p0/z, #32512
// CHECK-INST: mov     z21.s, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f 90 05  <unknown>

cpy     z21.d, p0/z, #-128
// CHECK-INST: mov     z21.d, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 10 d0 05  <unknown>

cpy     z21.d, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.d, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 d0 05  <unknown>

cpy     z21.d, p0/z, #-32768
// CHECK-INST: mov     z21.d, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 d0 05  <unknown>

cpy     z21.d, p0/z, #127
// CHECK-INST: mov     z21.d, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 0f d0 05  <unknown>

cpy     z21.d, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.d, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f d0 05  <unknown>

cpy     z21.d, p0/z, #32512
// CHECK-INST: mov     z21.d, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f d0 05  <unknown>


// --------------------------------------------------------------------------//
// Tests for merging variant (/m) and testing the range of predicate (> 7)
// is allowed.

cpy     z5.b, p15/m, #-128
// CHECK-INST: mov     z5.b, p15/m, #-128
// CHECK-ENCODING: [0x05,0x50,0x1f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 50 1f 05  <unknown>

cpy     z21.h, p15/m, #-128
// CHECK-INST: mov     z21.h, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0x5f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 50 5f 05  <unknown>

cpy     z21.h, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.h, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0x5f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 70 5f 05  <unknown>

cpy     z21.s, p15/m, #-128
// CHECK-INST: mov     z21.s, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0x9f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 50 9f 05  <unknown>

cpy     z21.s, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.s, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0x9f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 70 9f 05  <unknown>

cpy     z21.d, p15/m, #-128
// CHECK-INST: mov     z21.d, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 50 df 05  <unknown>

cpy     z21.d, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.d, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 70 df 05  <unknown>
