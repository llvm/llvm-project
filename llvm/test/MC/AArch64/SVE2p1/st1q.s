// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p1 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


st1q    {z0.q}, p0, [z0.d, x0]  // 11100100-00100000-00100000-00000000
// CHECK-INST: st1q    { z0.q }, p0, [z0.d, x0]
// CHECK-ENCODING: [0x00,0x20,0x20,0xe4]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e4202000 <unknown>

st1q    {z21.q}, p5, [z10.d, x21]  // 11100100-00110101-00110101-01010101
// CHECK-INST: st1q    { z21.q }, p5, [z10.d, x21]
// CHECK-ENCODING: [0x55,0x35,0x35,0xe4]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e4353555 <unknown>

st1q    {z23.q}, p3, [z13.d, x8]  // 11100100-00101000-00101101-10110111
// CHECK-INST: st1q    { z23.q }, p3, [z13.d, x8]
// CHECK-ENCODING: [0xb7,0x2d,0x28,0xe4]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e4282db7 <unknown>

st1q    {z31.q}, p7, [z31.d]  // 11100100-00111111-00111111-11111111
// CHECK-INST: st1q    { z31.q }, p7, [z31.d]
// CHECK-ENCODING: [0xff,0x3f,0x3f,0xe4]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e43f3fff <unknown>

