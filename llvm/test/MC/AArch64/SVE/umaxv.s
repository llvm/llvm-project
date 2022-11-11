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

umaxv b0, p7, z31.b
// CHECK-INST: umaxv	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x09,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04093fe0 <unknown>

umaxv h0, p7, z31.h
// CHECK-INST: umaxv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x49,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04493fe0 <unknown>

umaxv s0, p7, z31.s
// CHECK-INST: umaxv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x89,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04893fe0 <unknown>

umaxv d0, p7, z31.d
// CHECK-INST: umaxv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xc9,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c93fe0 <unknown>
