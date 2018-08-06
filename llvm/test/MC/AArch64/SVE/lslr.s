// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

lslr    z0.b, p0/m, z0.b, z0.b
// CHECK-INST: lslr	z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x80,0x17,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 17 04 <unknown>

lslr    z0.h, p0/m, z0.h, z0.h
// CHECK-INST: lslr	z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x57,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 57 04 <unknown>

lslr    z0.s, p0/m, z0.s, z0.s
// CHECK-INST: lslr	z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x80,0x97,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 97 04 <unknown>

lslr    z0.d, p0/m, z0.d, z0.d
// CHECK-INST: lslr	z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x80,0xd7,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 d7 04 <unknown>
