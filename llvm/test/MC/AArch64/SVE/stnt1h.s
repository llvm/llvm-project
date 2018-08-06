// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

stnt1h  z0.h, p0, [x0]
// CHECK-INST: stnt1h  { z0.h }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x90,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 90 e4 <unknown>

stnt1h  { z0.h }, p0, [x0]
// CHECK-INST: stnt1h  { z0.h }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x90,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 90 e4 <unknown>

stnt1h  { z23.h }, p3, [x13, #-8, mul vl]
// CHECK-INST: stnt1h  { z23.h }, p3, [x13, #-8, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x98,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 ed 98 e4 <unknown>

stnt1h  { z21.h }, p5, [x10, #7, mul vl]
// CHECK-INST: stnt1h  { z21.h }, p5, [x10, #7, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x97,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 f5 97 e4 <unknown>

stnt1h  { z0.h }, p0, [x0, x0, lsl #1]
// CHECK-INST: stnt1h  { z0.h }, p0, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0x80,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 80 e4 <unknown>
