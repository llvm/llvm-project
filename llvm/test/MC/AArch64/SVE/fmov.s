// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fmov z0.h, #0.0
// CHECK-INST: mov     z0.h, #0
// CHECK-ENCODING: [0x00,0xc0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 78 25

fmov z0.s, #0.0
// CHECK-INST: mov     z0.s, #0
// CHECK-ENCODING: [0x00,0xc0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 b8 25

fmov z0.d, #0.0
// CHECK-INST: mov     z0.d, #0
// CHECK-ENCODING: [0x00,0xc0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 f8 25

fmov z0.h, #-0.12500000
// CHECK-INST: fmov z0.h, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0x79,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 d8 79 25 <unknown>

fmov z0.s, #-0.12500000
// CHECK-INST: fmov z0.s, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0xb9,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 d8 b9 25 <unknown>

fmov z0.d, #-0.12500000
// CHECK-INST: fmov z0.d, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 d8 f9 25 <unknown>

fmov z0.d, #31.00000000
// CHECK-INST: fmov z0.d, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c7 f9 25 <unknown>
