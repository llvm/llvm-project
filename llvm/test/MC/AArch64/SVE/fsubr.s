// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fsubr   z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fsubr	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x43,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f 43 65 <unknown>

fsubr   z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fsubr	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x83,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f 83 65 <unknown>

fsubr   z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fsubr	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc3,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f c3 65 <unknown>
