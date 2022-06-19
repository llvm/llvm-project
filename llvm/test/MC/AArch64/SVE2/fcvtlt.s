// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN


fcvtlt z0.s, p0/m, z1.h
// CHECK-INST: fcvtlt z0.s, p0/m, z1.h
// CHECK-ENCODING: [0x20,0xa0,0x89,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 a0 89 64 <unknown>

fcvtlt z30.d, p7/m, z31.s
// CHECK-INST: fcvtlt z30.d, p7/m, z31.s
// CHECK-ENCODING: [0xfe,0xbf,0xcb,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: fe bf cb 64 <unknown>
