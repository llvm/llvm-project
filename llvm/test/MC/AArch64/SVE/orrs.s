// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

orrs    p0.b, p0/z, p0.b, p1.b
// CHECK-INST: orrs    p0.b, p0/z, p0.b, p1.b
// CHECK-ENCODING: [0x00,0x40,0xc1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 40 c1 25 <unknown>

orrs    p0.b, p0/z, p0.b, p0.b
// CHECK-INST: movs    p0.b, p0.b
// CHECK-ENCODING: [0x00,0x40,0xc0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 40 c0 25 <unknown>

orrs    p15.b, p15/z, p15.b, p15.b
// CHECK-INST: movs    p15.b, p15.b
// CHECK-ENCODING: [0xef,0x7d,0xcf,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ef 7d cf 25 <unknown>
