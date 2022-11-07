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


ld1q    {z0.q}, p0/z, [z0.d, x0]  // 11000100-00000000-10100000-00000000
// CHECK-INST: ld1q    { z0.q }, p0/z, [z0.d, x0]
// CHECK-ENCODING: [0x00,0xa0,0x00,0xc4]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: c400a000 <unknown>

ld1q    {z21.q}, p5/z, [z10.d, x21]  // 11000100-00010101-10110101-01010101
// CHECK-INST: ld1q    { z21.q }, p5/z, [z10.d, x21]
// CHECK-ENCODING: [0x55,0xb5,0x15,0xc4]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: c415b555 <unknown>

ld1q    {z23.q}, p3/z, [z13.d, x8]  // 11000100-00001000-10101101-10110111
// CHECK-INST: ld1q    { z23.q }, p3/z, [z13.d, x8]
// CHECK-ENCODING: [0xb7,0xad,0x08,0xc4]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: c408adb7 <unknown>

ld1q    {z31.q}, p7/z, [z31.d]  // 11000100-00011111-10111111-11111111
// CHECK-INST: ld1q    { z31.q }, p7/z, [z31.d]
// CHECK-ENCODING: [0xff,0xbf,0x1f,0xc4]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: c41fbfff <unknown>

