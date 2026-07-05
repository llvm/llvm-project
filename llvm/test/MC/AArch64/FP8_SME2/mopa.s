// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f8f16,+sme-f8f32 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f8f16,+sme-f8f32 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-f8f16,+sme-f8f32 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f8f16,+sme-f8f32 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-f8f16,-sme-f8f32 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f8f16,+sme-f8f32 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-f8f16,+sme-f8f32 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmopa   za0.h, p0/m, p0/m, z0.b, z0.b  // 10000000-10100000-00000000-00001000
// CHECK-INST: fmopa   za0.h, p0/m, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x08,0x00,0xa0,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: 80a00008 <unknown>


fmopa   za1.h, p7/m, p7/m, z31.b, z31.b  // 10000000-10111111-11111111-11101001
// CHECK-INST: fmopa   za1.h, p7/m, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xe9,0xff,0xbf,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: 80bfffe9 <unknown>


fmopa   za0.s, p0/m, p0/m, z0.b, z0.b  // 10000000-10100000-00000000-00000000
// CHECK-INST: fmopa   za0.s, p0/m, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0xa0,0x80]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: 80a00000 <unknown>

fmopa   za3.s, p7/m, p7/m, z31.b, z31.b  // 10000000-10111111-11111111-11100011
// CHECK-INST: fmopa   za3.s, p7/m, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xe3,0xff,0xbf,0x80]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: 80bfffe3 <unknown>
