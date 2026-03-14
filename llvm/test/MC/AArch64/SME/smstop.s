// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-INST

smstop
// CHECK-INST: smstop
// CHECK-ENCODING: [0x7f,0x46,0x03,0xd5]

smstop sm
// CHECK-INST: smstop sm
// CHECK-ENCODING: [0x7f,0x42,0x03,0xd5]

smstop za
// CHECK-INST: smstop za
// CHECK-ENCODING: [0x7f,0x44,0x03,0xd5]

smstop SM
// CHECK-INST: smstop sm
// CHECK-ENCODING: [0x7f,0x42,0x03,0xd5]

smstop ZA
// CHECK-INST: smstop za
// CHECK-ENCODING: [0x7f,0x44,0x03,0xd5]
