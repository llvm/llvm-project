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

fminnmv h0, p7, z31.h
// CHECK-INST: fminnmv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x45,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65453fe0 <unknown>

fminnmv s0, p7, z31.s
// CHECK-INST: fminnmv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x85,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65853fe0 <unknown>

fminnmv d0, p7, z31.d
// CHECK-INST: fminnmv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xc5,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c53fe0 <unknown>
