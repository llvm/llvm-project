// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

rev   z0.b, z31.b
// CHECK-INST: rev	z0.b, z31.b
// CHECK-ENCODING: [0xe0,0x3b,0x38,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3b 38 05 <unknown>

rev   z0.h, z31.h
// CHECK-INST: rev	z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x3b,0x78,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3b 78 05 <unknown>

rev   z0.s, z31.s
// CHECK-INST: rev	z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x3b,0xb8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3b b8 05 <unknown>

rev   z0.d, z31.d
// CHECK-INST: rev	z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x3b,0xf8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3b f8 05 <unknown>
