// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3,+fpv2 < %s \
// RUN:        | FileCheck %s --ignore-case --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p3,+fpv2 < %s \
// RUN:        | FileCheck %s --ignore-case --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --ignore-case --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3,+fpv2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p3,+fpv2 --no-print-imm-hex - | FileCheck %s --ignore-case --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3,+fpv2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p3,-fpv2 --no-print-imm-hex - | FileCheck %s --ignore-case --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3,+fpv2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p3,+fpv2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --ignore-case --check-prefixes=CHECK-ENCODING,CHECK-INST

fcmtl p15.h, z31.h, z31.h
// CHECK-INST: fcmtl p15.h, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xef,0x13,0x5f,0x65]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 655f13ef <unknown>

fcmtl p15.s, z31.s, z31.s
// CHECK-INST: fcmtl p15.s, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xef,0x13,0x9f,0x65]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 659f13ef <unknown>

fcmtl p15.d, z31.d, z31.d
// CHECK-INST: fcmtl p15.d, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xef,0x13,0xdf,0x65]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 65df13ef <unknown>

fcmtlmg p15.h, z31.h, z31.h
// CHECK-INST: fcmtlmg p15.h, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xef,0x17,0x5f,0x65]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 655f17ef <unknown>

fcmtlmg p15.s, z31.s, z31.s
// CHECK-INST: fcmtlmg p15.s, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xef,0x17,0x9f,0x65]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 659f17ef <unknown>

fcmtlmg p15.d, z31.d, z31.d
// CHECK-INST: fcmtlmg p15.d, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xef,0x17,0xdf,0x65]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 65df17ef <unknown>
