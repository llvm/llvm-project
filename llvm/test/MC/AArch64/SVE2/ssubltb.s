// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


ssubltb z0.h, z1.b, z31.b
// CHECK-INST: ssubltb	z0.h, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x8c,0x5f,0x45]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 8c 5f 45 <unknown>

ssubltb z0.s, z1.h, z31.h
// CHECK-INST: ssubltb	z0.s, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x8c,0x9f,0x45]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 8c 9f 45 <unknown>

ssubltb z0.d, z1.s, z31.s
// CHECK-INST: ssubltb	z0.d, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x8c,0xdf,0x45]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 8c df 45 <unknown>
