// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fminv h0, p7, z31.h
// CHECK-INST: fminv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x47,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 47 65 <unknown>

fminv s0, p7, z31.s
// CHECK-INST: fminv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x87,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 87 65 <unknown>

fminv d0, p7, z31.d
// CHECK-INST: fminv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xc7,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f c7 65 <unknown>
