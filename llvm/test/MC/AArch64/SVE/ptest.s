// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ptest p15, p0.b
// CHECK-INST: ptest	p15, p0.b
// CHECK-ENCODING: [0x00,0xfc,0x50,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 fc 50 25 <unknown>

ptest p15, p15.b
// CHECK-INST: ptest	p15, p15.b
// CHECK-ENCODING: [0xe0,0xfd,0x50,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fd 50 25 <unknown>
