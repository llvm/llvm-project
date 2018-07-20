// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

addpl   x21, x21, #0
// CHECK-INST: addpl   x21, x21, #0
// CHECK-ENCODING: [0x15,0x50,0x75,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 50 75 04 <unknown>

addpl   x23, x8, #-1
// CHECK-INST: addpl   x23, x8, #-1
// CHECK-ENCODING: [0xf7,0x57,0x68,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f7 57 68 04 <unknown>

addpl   sp, sp, #31
// CHECK-INST: addpl   sp, sp, #31
// CHECK-ENCODING: [0xff,0x53,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 53 7f 04 <unknown>

addpl   x0, x0, #-32
// CHECK-INST: addpl   x0, x0, #-32
// CHECK-ENCODING: [0x00,0x54,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 54 60 04 <unknown>
