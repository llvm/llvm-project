// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

brka  p0.b, p15/m, p15.b
// CHECK-INST: brka	p0.b, p15/m, p15.b
// CHECK-ENCODING: [0xf0,0x7d,0x10,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f0 7d 10 25 <unknown>

brka  p0.b, p15/z, p15.b
// CHECK-INST: brka	p0.b, p15/z, p15.b
// CHECK-ENCODING: [0xe0,0x7d,0x10,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 7d 10 25 <unknown>
