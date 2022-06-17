// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN

rdsvl    x0, #0
// CHECK-INST: rdsvl    x0, #0
// CHECK-ENCODING: [0x00,0x58,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 58 bf 04 <unknown>

rdsvl    xzr, #-1
// CHECK-INST: rdsvl    xzr, #-1
// CHECK-ENCODING: [0xff,0x5f,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 5f bf 04 <unknown>

rdsvl    x23, #31
// CHECK-INST: rdsvl    x23, #31
// CHECK-ENCODING: [0xf7,0x5b,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: f7 5b bf 04 <unknown>

rdsvl    x21, #-32
// CHECK-INST: rdsvl    x21, #-32
// CHECK-ENCODING: [0x15,0x5c,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 15 5c bf 04 <unknown>
