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


uunpk   {z0.h - z1.h}, z0.b  // 11000001-01100101-11100000-00000001
// CHECK-INST: uunpk   { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x01,0xe0,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165e001 <unknown>

uunpk   {z20.h - z21.h}, z10.b  // 11000001-01100101-11100001-01010101
// CHECK-INST: uunpk   { z20.h, z21.h }, z10.b
// CHECK-ENCODING: [0x55,0xe1,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165e155 <unknown>

uunpk   {z22.h - z23.h}, z13.b  // 11000001-01100101-11100001-10110111
// CHECK-INST: uunpk   { z22.h, z23.h }, z13.b
// CHECK-ENCODING: [0xb7,0xe1,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165e1b7 <unknown>

uunpk   {z30.h - z31.h}, z31.b  // 11000001-01100101-11100011-11111111
// CHECK-INST: uunpk   { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xff,0xe3,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165e3ff <unknown>


uunpk   {z0.s - z1.s}, z0.h  // 11000001-10100101-11100000-00000001
// CHECK-INST: uunpk   { z0.s, z1.s }, z0.h
// CHECK-ENCODING: [0x01,0xe0,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5e001 <unknown>

uunpk   {z20.s - z21.s}, z10.h  // 11000001-10100101-11100001-01010101
// CHECK-INST: uunpk   { z20.s, z21.s }, z10.h
// CHECK-ENCODING: [0x55,0xe1,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5e155 <unknown>

uunpk   {z22.s - z23.s}, z13.h  // 11000001-10100101-11100001-10110111
// CHECK-INST: uunpk   { z22.s, z23.s }, z13.h
// CHECK-ENCODING: [0xb7,0xe1,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5e1b7 <unknown>

uunpk   {z30.s - z31.s}, z31.h  // 11000001-10100101-11100011-11111111
// CHECK-INST: uunpk   { z30.s, z31.s }, z31.h
// CHECK-ENCODING: [0xff,0xe3,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5e3ff <unknown>


uunpk   {z0.d - z1.d}, z0.s  // 11000001-11100101-11100000-00000001
// CHECK-INST: uunpk   { z0.d, z1.d }, z0.s
// CHECK-ENCODING: [0x01,0xe0,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5e001 <unknown>

uunpk   {z20.d - z21.d}, z10.s  // 11000001-11100101-11100001-01010101
// CHECK-INST: uunpk   { z20.d, z21.d }, z10.s
// CHECK-ENCODING: [0x55,0xe1,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5e155 <unknown>

uunpk   {z22.d - z23.d}, z13.s  // 11000001-11100101-11100001-10110111
// CHECK-INST: uunpk   { z22.d, z23.d }, z13.s
// CHECK-ENCODING: [0xb7,0xe1,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5e1b7 <unknown>

uunpk   {z30.d - z31.d}, z31.s  // 11000001-11100101-11100011-11111111
// CHECK-INST: uunpk   { z30.d, z31.d }, z31.s
// CHECK-ENCODING: [0xff,0xe3,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5e3ff <unknown>


uunpk   {z0.h - z3.h}, {z0.b - z1.b}  // 11000001-01110101-11100000-00000001
// CHECK-INST: uunpk   { z0.h - z3.h }, { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0xe0,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175e001 <unknown>

uunpk   {z20.h - z23.h}, {z10.b - z11.b}  // 11000001-01110101-11100001-01010101
// CHECK-INST: uunpk   { z20.h - z23.h }, { z10.b, z11.b }
// CHECK-ENCODING: [0x55,0xe1,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175e155 <unknown>

uunpk   {z20.h - z23.h}, {z12.b - z13.b}  // 11000001-01110101-11100001-10010101
// CHECK-INST: uunpk   { z20.h - z23.h }, { z12.b, z13.b }
// CHECK-ENCODING: [0x95,0xe1,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175e195 <unknown>

uunpk   {z28.h - z31.h}, {z30.b - z31.b}  // 11000001-01110101-11100011-11011101
// CHECK-INST: uunpk   { z28.h - z31.h }, { z30.b, z31.b }
// CHECK-ENCODING: [0xdd,0xe3,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175e3dd <unknown>


uunpk   {z0.s - z3.s}, {z0.h - z1.h}  // 11000001-10110101-11100000-00000001
// CHECK-INST: uunpk   { z0.s - z3.s }, { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0xe0,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5e001 <unknown>

uunpk   {z20.s - z23.s}, {z10.h - z11.h}  // 11000001-10110101-11100001-01010101
// CHECK-INST: uunpk   { z20.s - z23.s }, { z10.h, z11.h }
// CHECK-ENCODING: [0x55,0xe1,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5e155 <unknown>

uunpk   {z20.s - z23.s}, {z12.h - z13.h}  // 11000001-10110101-11100001-10010101
// CHECK-INST: uunpk   { z20.s - z23.s }, { z12.h, z13.h }
// CHECK-ENCODING: [0x95,0xe1,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5e195 <unknown>

uunpk   {z28.s - z31.s}, {z30.h - z31.h}  // 11000001-10110101-11100011-11011101
// CHECK-INST: uunpk   { z28.s - z31.s }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdd,0xe3,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5e3dd <unknown>


uunpk   {z0.d - z3.d}, {z0.s - z1.s}  // 11000001-11110101-11100000-00000001
// CHECK-INST: uunpk   { z0.d - z3.d }, { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0xe0,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5e001 <unknown>

uunpk   {z20.d - z23.d}, {z10.s - z11.s}  // 11000001-11110101-11100001-01010101
// CHECK-INST: uunpk   { z20.d - z23.d }, { z10.s, z11.s }
// CHECK-ENCODING: [0x55,0xe1,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5e155 <unknown>

uunpk   {z20.d - z23.d}, {z12.s - z13.s}  // 11000001-11110101-11100001-10010101
// CHECK-INST: uunpk   { z20.d - z23.d }, { z12.s, z13.s }
// CHECK-ENCODING: [0x95,0xe1,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5e195 <unknown>

uunpk   {z28.d - z31.d}, {z30.s - z31.s}  // 11000001-11110101-11100011-11011101
// CHECK-INST: uunpk   { z28.d - z31.d }, { z30.s, z31.s }
// CHECK-ENCODING: [0xdd,0xe3,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5e3dd <unknown>

