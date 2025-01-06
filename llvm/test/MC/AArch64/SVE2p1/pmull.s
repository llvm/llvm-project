// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-aes2,+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-aes2,+ssve-aes < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve-aes2,+sve2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve-aes2,+sve2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve-aes2,+sve2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve-aes2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-aes2,+sve2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve-aes2,+sve2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

pmull   {z0.q-z1.q}, z0.d, z0.d  // 01000101-00100000-11111000-00000000
// CHECK-INST: pmull   { z0.q, z1.q }, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xf8,0x20,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4520f800 <unknown>

pmull   {z22.q-z23.q}, z13.d, z8.d  // 01000101-00101000-11111001-10110110
// CHECK-INST: pmull   { z22.q, z23.q }, z13.d, z8.d
// CHECK-ENCODING: [0xb6,0xf9,0x28,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4528f9b6 <unknown>

pmull   {z30.q-z31.q}, z31.d, z31.d  // 01000101-00111111-11111011-11111110
// CHECK-INST: pmull   { z30.q, z31.q }, z31.d, z31.d
// CHECK-ENCODING: [0xfe,0xfb,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 453ffbfe <unknown>