// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

punpkhi p0.h, p0.b
// CHECK-INST: punpkhi	p0.h, p0.b
// CHECK-ENCODING: [0x00,0x40,0x31,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 31 05 <unknown>

punpkhi p15.h, p15.b
// CHECK-INST: punpkhi	p15.h, p15.b
// CHECK-ENCODING: [0xef,0x41,0x31,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 41 31 05 <unknown>
