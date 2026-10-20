// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+sve-lrcpc < %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+sve-lrcpc < %s \
// RUN:   | llvm-objdump -d --mattr=+sve,+sve-lrcpc --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+sve-lrcpc < %s \
// RUN:   | llvm-objdump -d --mattr=+sve,-sve-lrcpc --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+sve-lrcpc < %s \
// RUN:   | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:   | llvm-mc -triple=aarch64 -mattr=+sve,+sve-lrcpc -disassemble -show-encoding \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

ldapro z0, [x0]
// CHECK-INST: ldapro z0, [x0]
// CHECK-ENCODING: encoding: [0x00,0x60,0x80,0x85]
// CHECK-ERROR: instruction requires: sve-lrcpc
// CHECK-UNKNOWN: 85806000 <unknown>

ldapro z31, [x0, #0]
// CHECK-INST: ldapro z31, [x0]
// CHECK-ENCODING: encoding: [0x1f,0x60,0x80,0x85]
// CHECK-ERROR: instruction requires: sve-lrcpc
// CHECK-UNKNOWN: 8580601f <unknown>

ldapro z0, [sp, #2720]
// CHECK-INST: ldapro z0, [sp, #2720]
// CHECK-ENCODING: encoding: [0xe0,0x77,0x8a,0x85]
// CHECK-ERROR: instruction requires: sve-lrcpc
// CHECK-UNKNOWN: 858a77e0 <unknown>

ldapro z31, [sp, #8160]
// CHECK-INST: ldapro z31, [sp, #8160]
// CHECK-ENCODING: encoding: [0xff,0x7f,0x9f,0x85]
// CHECK-ERROR: instruction requires: sve-lrcpc
// CHECK-UNKNOWN: 859f7fff <unknown>

stlro z0, [x0]
// CHECK-INST: stlro z0, [x0]
// CHECK-ENCODING: encoding: [0x00,0x00,0xc0,0xe5]
// CHECK-ERROR: instruction requires: sve-lrcpc
// CHECK-UNKNOWN: e5c00000 <unknown>

stlro z31, [x0, #0]
// CHECK-INST: stlro z31, [x0]
// CHECK-ENCODING: encoding: [0x1f,0x00,0xc0,0xe5]
// CHECK-ERROR: instruction requires: sve-lrcpc
// CHECK-UNKNOWN: e5c0001f <unknown>

stlro z0, [sp, #2720]
// CHECK-INST: stlro z0, [sp, #2720]
// CHECK-ENCODING: encoding: [0xe0,0x17,0xca,0xe5]
// CHECK-ERROR: instruction requires: sve-lrcpc
// CHECK-UNKNOWN: e5ca17e0 <unknown>

stlro z31, [sp, #8160]
// CHECK-INST: stlro z31, [sp, #8160]
// CHECK-ENCODING: encoding: [0xff,0x1f,0xdf,0xe5]
// CHECK-ERROR: instruction requires: sve-lrcpc
// CHECK-UNKNOWN: e5df1fff <unknown>
