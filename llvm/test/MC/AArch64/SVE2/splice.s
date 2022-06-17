// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

splice  z29.b, p7, { z30.b, z31.b }
// CHECK-INST: splice  z29.b, p7, { z30.b, z31.b }
// CHECK-ENCODING: [0xdd,0x9f,0x2d,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 9f 2d 05 <unknown>

splice  z29.h, p7, { z30.h, z31.h }
// CHECK-INST: splice  z29.h, p7, { z30.h, z31.h }
// CHECK-ENCODING: [0xdd,0x9f,0x6d,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 9f 6d 05 <unknown>

splice  z29.s, p7, { z30.s, z31.s }
// CHECK-INST: splice  z29.s, p7, { z30.s, z31.s }
// CHECK-ENCODING: [0xdd,0x9f,0xad,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 9f ad 05 <unknown>

splice  z29.d, p7, { z30.d, z31.d }
// CHECK-INST: splice  z29.d, p7, { z30.d, z31.d }
// CHECK-ENCODING: [0xdd,0x9f,0xed,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 9f ed 05 <unknown>
