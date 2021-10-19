// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

splice  z31.b, p7, z31.b, z31.b
// CHECK-INST: splice  z31.b, p7, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x9f,0x2c,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 9f 2c 05 <unknown>

splice  z31.h, p7, z31.h, z31.h
// CHECK-INST: splice  z31.h, p7, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x9f,0x6c,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 9f 6c 05 <unknown>

splice  z31.s, p7, z31.s, z31.s
// CHECK-INST: splice  z31.s, p7, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x9f,0xac,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 9f ac 05 <unknown>

splice  z31.d, p7, z31.d, z31.d
// CHECK-INST: splice  z31.d, p7, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x9f,0xec,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 9f ec 05 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

splice  z4.d, p7, z4.d, z31.d
// CHECK-INST: splice	z4.d, p7, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x9f,0xec,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e4 9f ec 05 <unknown>
