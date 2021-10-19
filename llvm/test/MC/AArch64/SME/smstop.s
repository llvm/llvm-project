// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

smstop
// CHECK-INST: smstop
// CHECK-ENCODING: [0x7f,0x46,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 46 03 d5   msr   S0_3_C4_C6_3, xzr

smstop sm
// CHECK-INST: smstop sm
// CHECK-ENCODING: [0x7f,0x42,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 42 03 d5   msr   S0_3_C4_C2_3, xzr

smstop za
// CHECK-INST: smstop za
// CHECK-ENCODING: [0x7f,0x44,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 44 03 d5   msr   S0_3_C4_C4_3, xzr

smstop SM
// CHECK-INST: smstop sm
// CHECK-ENCODING: [0x7f,0x42,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 42 03 d5   msr   S0_3_C4_C2_3, xzr

smstop ZA
// CHECK-INST: smstop za
// CHECK-ENCODING: [0x7f,0x44,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 44 03 d5   msr   S0_3_C4_C4_3, xzr
