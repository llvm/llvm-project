// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fmaxv h0, p7, z31.h
// CHECK-INST: fmaxv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x46,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 46 65 <unknown>

fmaxv s0, p7, z31.s
// CHECK-INST: fmaxv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x86,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 86 65 <unknown>

fmaxv d0, p7, z31.d
// CHECK-INST: fmaxv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xc6,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f c6 65 <unknown>
