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


usublt z0.h, z1.b, z2.b
// CHECK-INST: usublt z0.h, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x1c,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 1c 42 45 <unknown>

usublt z29.s, z30.h, z31.h
// CHECK-INST: usublt z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x1f,0x9f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 1f 9f 45 <unknown>

usublt z31.d, z31.s, z31.s
// CHECK-INST: usublt z31.d, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 1f df 45 <unknown>
