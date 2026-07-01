// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+hinte < %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-INST,CHECK-ENCODING
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+hinte \
// RUN:   < %S/Inputs/hinte-generic-sysreg.s \
// RUN:   | FileCheck %S/Inputs/hinte-generic-sysreg.s --check-prefix=CHECK
// RUN: not llvm-mc -triple=aarch64 -show-encoding \
// RUN:   < %S/Inputs/hinte-generic-sysreg.s 2>&1 \
// RUN:   | FileCheck %S/Inputs/hinte-generic-sysreg.s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+hinte < %s \
// RUN:   | llvm-objdump -d --mattr=+hinte --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+hinte < %s \
// RUN:   | llvm-objdump -d --mattr=-hinte --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-UNKNOWN

hinte #0
// CHECK-INST: hinte #0
// CHECK-ENCODING: encoding: [0x00,0x20,0x00,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-UNKNOWN: d5002000 <unknown>

hinte #16352
// CHECK-INST: hinte #16352
// CHECK-ENCODING: encoding: [0xe0,0x2f,0x03,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-UNKNOWN: d5032fe0 <unknown>

hinte #16353
// CHECK-INST: hinte #16353
// CHECK-ENCODING: encoding: [0xe1,0x2f,0x03,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-UNKNOWN: d5032fe1 <unknown>

hinte #21845
// CHECK-INST: hinte #21845
// CHECK-ENCODING: encoding: [0x55,0x25,0x05,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-UNKNOWN: d5052555 <unknown>

hinte #43690
// CHECK-INST: hinte #43690
// CHECK-ENCODING: encoding: [0xaa,0x2a,0x22,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-UNKNOWN: d5222aaa <unknown>

hinte #65535
// CHECK-INST: hinte #65535
// CHECK-ENCODING: encoding: [0xff,0x2f,0x27,0xd5]
// CHECK-ERROR: instruction requires: hinte
// CHECK-UNKNOWN: d5272fff <unknown>
