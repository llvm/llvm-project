// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

addsvl   x21, x21, #0
// CHECK-INST: addsvl   x21, x21, #0
// CHECK-ENCODING: [0x15,0x58,0x35,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 15 58 35 04 <unknown>

addsvl   x23, x8, #-1
// CHECK-INST: addsvl   x23, x8, #-1
// CHECK-ENCODING: [0xf7,0x5f,0x28,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: f7 5f 28 04 <unknown>

addsvl   sp, sp, #31
// CHECK-INST: addsvl   sp, sp, #31
// CHECK-ENCODING: [0xff,0x5b,0x3f,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 5b 3f 04 <unknown>

addsvl   x0, x0, #-32
// CHECK-INST: addsvl   x0, x0, #-32
// CHECK-ENCODING: [0x00,0x5c,0x20,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 5c 20 04 <unknown>
