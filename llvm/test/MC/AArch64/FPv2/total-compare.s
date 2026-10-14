// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fpv2 < %s \
// RUN:        | FileCheck %s --ignore-case --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --ignore-case --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fpv2 < %s \
// RUN:        | llvm-objdump -d --mattr=+fpv2 --no-print-imm-hex - | FileCheck %s --ignore-case --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fpv2 < %s \
// RUN:        | llvm-objdump -d --mattr=-fpv2 --no-print-imm-hex - | FileCheck %s --ignore-case --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fpv2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+fpv2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --ignore-case --check-prefixes=CHECK-ENCODING,CHECK-INST

fcmtl h31, h0
// CHECK-INST: fcmtl h31, h0
// CHECK-ENCODING: encoding: [0xe0,0xa3,0xe0,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1ee0a3e0 <unknown>

fcmtl s31, s0
// CHECK-INST: fcmtl s31, s0
// CHECK-ENCODING: encoding: [0xe0,0xa3,0x20,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e20a3e0 <unknown>

fcmtl d31, d0
// CHECK-INST: fcmtl d31, d0
// CHECK-ENCODING: encoding: [0xe0,0xa3,0x60,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e60a3e0 <unknown>

fcmtlmg h31, h0
// CHECK-INST: fcmtlmg h31, h0
// CHECK-ENCODING: encoding: [0xe8,0xa3,0xe0,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1ee0a3e8 <unknown>

fcmtlmg s31, s0
// CHECK-INST: fcmtlmg s31, s0
// CHECK-ENCODING: encoding: [0xe8,0xa3,0x20,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e20a3e8 <unknown>

fcmtlmg d31, d0
// CHECK-INST: fcmtlmg d31, d0
// CHECK-ENCODING: encoding: [0xe8,0xa3,0x60,0x1e]
// CHECK-ERROR: instruction requires: fpv2
// CHECK-UNKNOWN: 1e60a3e8 <unknown>
