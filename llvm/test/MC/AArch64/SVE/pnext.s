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

pnext p15.b, p15, p15.b
// CHECK-INST: pnext	p15.b, p15, p15.b
// CHECK-ENCODING: [0xef,0xc5,0x19,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ef c5 19 25 <unknown>

pnext p0.b, p15, p0.b
// CHECK-INST: pnext	p0.b, p15, p0.b
// CHECK-ENCODING: [0xe0,0xc5,0x19,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c5 19 25 <unknown>

pnext p0.h, p15, p0.h
// CHECK-INST: pnext	p0.h, p15, p0.h
// CHECK-ENCODING: [0xe0,0xc5,0x59,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c5 59 25 <unknown>

pnext p0.s, p15, p0.s
// CHECK-INST: pnext	p0.s, p15, p0.s
// CHECK-ENCODING: [0xe0,0xc5,0x99,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c5 99 25 <unknown>

pnext p0.d, p15, p0.d
// CHECK-INST: pnext	p0.d, p15, p0.d
// CHECK-ENCODING: [0xe0,0xc5,0xd9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c5 d9 25 <unknown>
