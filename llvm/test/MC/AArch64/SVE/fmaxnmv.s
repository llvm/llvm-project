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

fmaxnmv h0, p7, z31.h
// CHECK-INST: fmaxnmv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x44,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f 44 65 <unknown>

fmaxnmv s0, p7, z31.s
// CHECK-INST: fmaxnmv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x84,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f 84 65 <unknown>

fmaxnmv d0, p7, z31.d
// CHECK-INST: fmaxnmv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xc4,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f c4 65 <unknown>
