// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f8f16 < %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f16f8f16 < %s \
// RUN:   | llvm-objdump -d --mattr=+sme-f16f8f16 --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f16f8f16 < %s \
// RUN:   | llvm-objdump -d --mattr=-sme-f16f8f16 --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f8f16 < %s \
// RUN:   | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:   | llvm-mc -triple=aarch64 -mattr=+sme-f16f8f16 -disassemble -show-encoding \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-INST: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-ENCODING: encoding: [0x00,0x00,0x60,0xc0]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: c0600000 <unknown>

fdot za.h[w8, 7], { z30.b, z31.b }, z15.h[3], z23.b
// CHECK-INST: fdot za.h[w8, 7, vgx2], { z30.b, z31.b }, z15.h[3], z23.b
// CHECK-ENCODING: encoding: [0xdf,0x0f,0x6f,0xc0]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: c06f0fdf <unknown>

fdot za.h[w11, 0, vgx2], { z0.b, z1.b }, z15.h[3], z28.b
// CHECK-INST: fdot za.h[w11, 0, vgx2], { z0.b, z1.b }, z15.h[3], z28.b
// CHECK-ENCODING: encoding: [0x18,0x70,0x6f,0xc0]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: c06f7018 <unknown>

fdot za.h[w11, 7], { z30.b, z31.b }, z0.h[0], z31.b
// CHECK-INST: fdot za.h[w11, 7, vgx2], { z30.b, z31.b }, z0.h[0], z31.b
// CHECK-ENCODING: encoding: [0xc7,0x7f,0x60,0xc0]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: c0607fc7 <unknown>

fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-INST: fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-ENCODING: encoding: [0x00,0x00,0xe0,0xc0]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: c0e00000 <unknown>

fvdot za.h[w8, 7], { z30.b, z31.b }, z15.h[3], z23.b
// CHECK-INST: fvdot za.h[w8, 7, vgx2], { z30.b, z31.b }, z15.h[3], z23.b
// CHECK-ENCODING: encoding: [0xdf,0x0f,0xef,0xc0]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: c0ef0fdf <unknown>

fvdot za.h[w11, 0, vgx2], { z0.b, z1.b }, z15.h[3], z28.b
// CHECK-INST: fvdot za.h[w11, 0, vgx2], { z0.b, z1.b }, z15.h[3], z28.b
// CHECK-ENCODING: encoding: [0x18,0x70,0xef,0xc0]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: c0ef7018 <unknown>

fvdot za.h[w11, 7], { z30.b, z31.b }, z0.h[0], z31.b
// CHECK-INST: fvdot za.h[w11, 7, vgx2], { z30.b, z31.b }, z0.h[0], z31.b
// CHECK-ENCODING: encoding: [0xc7,0x7f,0xe0,0xc0]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: c0e07fc7 <unknown>
