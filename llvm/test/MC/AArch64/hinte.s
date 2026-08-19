// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+hinte < %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-INST,CHECK-ASM,CHECK-ENCODING
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+hinte < %s \
// RUN:   | llvm-objdump -d --mattr=+hinte --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+hinte < %s \
// RUN:   | llvm-objdump -d --mattr=-hinte --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-GENERIC

hinte #0
// CHECK-INST: hinte #0
// CHECK-ENCODING: encoding: [0x00,0x20,0x00,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-GENERIC: msr S0_0_C2_C0_0, x0

hinte #16352
// CHECK-INST: hinte #16352
// CHECK-ENCODING: encoding: [0xe0,0x2f,0x03,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-GENERIC: msr S0_3_C2_C15_7, x0

hinte #16353
// CHECK-INST: hinte #16353
// CHECK-ENCODING: encoding: [0xe1,0x2f,0x03,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-GENERIC: msr S0_3_C2_C15_7, x1

hinte #21845
// CHECK-INST: hinte #21845
// CHECK-ENCODING: encoding: [0x55,0x25,0x05,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-GENERIC: msr S0_5_C2_C5_2, x21

hinte #43690
// CHECK-INST: hinte #43690
// CHECK-ENCODING: encoding: [0xaa,0x2a,0x22,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-GENERIC: mrs x10, S0_2_C2_C10_5

hinte #65535
// CHECK-INST: hinte #65535
// CHECK-ENCODING: encoding: [0xff,0x2f,0x27,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-GENERIC: mrs xzr, S0_7_C2_C15_7

msr S0_0_C2_C0_0, x0
// CHECK-ASM: msr S0_0_C2_C0_0, x0
// CHECK-ENCODING: encoding: [0x00,0x20,0x00,0xd5]
// CHECK-ERROR: msr S0_0_C2_C0_0, x0

mrs x0, S0_0_C2_C0_0
// CHECK-ASM: mrs x0, S0_0_C2_C0_0
// CHECK-ENCODING: encoding: [0x00,0x20,0x20,0xd5]
// CHECK-ERROR: mrs x0, S0_0_C2_C0_0
