// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN

rdsvl    x0, #0
// CHECK-INST: rdsvl    x0, #0
// CHECK-ENCODING: [0x00,0x58,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 04bf5800 <unknown>

rdsvl    xzr, #-1
// CHECK-INST: rdsvl    xzr, #-1
// CHECK-ENCODING: [0xff,0x5f,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 04bf5fff <unknown>

rdsvl    x23, #31
// CHECK-INST: rdsvl    x23, #31
// CHECK-ENCODING: [0xf7,0x5b,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 04bf5bf7 <unknown>

rdsvl    x21, #-32
// CHECK-INST: rdsvl    x21, #-32
// CHECK-ENCODING: [0x15,0x5c,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 04bf5c15 <unknown>
