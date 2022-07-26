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

frsqrts z0.h, z1.h, z31.h
// CHECK-INST: frsqrts	z0.h, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x1c,0x5f,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655f1c20 <unknown>

frsqrts z0.s, z1.s, z31.s
// CHECK-INST: frsqrts	z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x1c,0x9f,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659f1c20 <unknown>

frsqrts z0.d, z1.d, z31.d
// CHECK-INST: frsqrts	z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x1c,0xdf,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65df1c20 <unknown>
