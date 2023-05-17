// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ftsmul z0.h, z1.h, z31.h
// CHECK-INST: ftsmul	z0.h, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x0c,0x5f,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 655f0c20 <unknown>

ftsmul z0.s, z1.s, z31.s
// CHECK-INST: ftsmul	z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x0c,0x9f,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 659f0c20 <unknown>

ftsmul z0.d, z1.d, z31.d
// CHECK-INST: ftsmul	z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x0c,0xdf,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 65df0c20 <unknown>
