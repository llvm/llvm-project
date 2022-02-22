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

ext z0.b, { z1.b, z2.b }, #0
// CHECK-INST: ext z0.b, { z1.b, z2.b }, #0
// CHECK-ENCODING: [0x20,0x00,0x60,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 00 60 05 <unknown>

ext z31.b, { z30.b, z31.b }, #255
// CHECK-INST: ext z31.b, { z30.b, z31.b }, #255
// CHECK-ENCODING: [0xdf,0x1f,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 1f 7f 05 <unknown>
