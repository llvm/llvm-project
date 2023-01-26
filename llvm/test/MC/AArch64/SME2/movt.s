// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


movt    x0, zt0[0]  // 11000000-01001100-00000011-11100000
// CHECK-INST: movt    x0, zt0[0]
// CHECK-ENCODING: [0xe0,0x03,0x4c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04c03e0 <unknown>

movt    x21, zt0[40]  // 11000000-01001100-01010011-11110101
// CHECK-INST: movt    x21, zt0[40]
// CHECK-ENCODING: [0xf5,0x53,0x4c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04c53f5 <unknown>

movt    x23, zt0[48]  // 11000000-01001100-01100011-11110111
// CHECK-INST: movt    x23, zt0[48]
// CHECK-ENCODING: [0xf7,0x63,0x4c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04c63f7 <unknown>

movt    xzr, zt0[56]  // 11000000-01001100-01110011-11111111
// CHECK-INST: movt    xzr, zt0[56]
// CHECK-ENCODING: [0xff,0x73,0x4c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04c73ff <unknown>


movt    zt0[0], x0  // 11000000-01001110-00000011-11100000
// CHECK-INST: movt    zt0[0], x0
// CHECK-ENCODING: [0xe0,0x03,0x4e,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04e03e0 <unknown>

movt    zt0[40], x21  // 11000000-01001110-01010011-11110101
// CHECK-INST: movt    zt0[40], x21
// CHECK-ENCODING: [0xf5,0x53,0x4e,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04e53f5 <unknown>

movt    zt0[48], x23  // 11000000-01001110-01100011-11110111
// CHECK-INST: movt    zt0[48], x23
// CHECK-ENCODING: [0xf7,0x63,0x4e,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04e63f7 <unknown>

movt    zt0[56], xzr  // 11000000-01001110-01110011-11111111
// CHECK-INST: movt    zt0[56], xzr
// CHECK-ENCODING: [0xff,0x73,0x4e,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04e73ff <unknown>

