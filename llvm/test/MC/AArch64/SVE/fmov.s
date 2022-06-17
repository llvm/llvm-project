// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fmov z0.h, #0.0
// CHECK-INST: mov     z0.h, #0
// CHECK-ENCODING: [0x00,0xc0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 78 25

fmov z0.s, #0.0
// CHECK-INST: mov     z0.s, #0
// CHECK-ENCODING: [0x00,0xc0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 b8 25

fmov z0.d, #0.0
// CHECK-INST: mov     z0.d, #0
// CHECK-ENCODING: [0x00,0xc0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 f8 25

fmov z0.h, #-0.12500000
// CHECK-INST: fmov z0.h, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0x79,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d8 79 25 <unknown>

fmov z0.s, #-0.12500000
// CHECK-INST: fmov z0.s, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0xb9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d8 b9 25 <unknown>

fmov z0.d, #-0.12500000
// CHECK-INST: fmov z0.d, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d8 f9 25 <unknown>

fmov z0.d, #31.00000000
// CHECK-INST: fmov z0.d, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c7 f9 25 <unknown>

fmov z0.h, p0/m, #-0.12500000
// CHECK-INST: fmov z0.h, p0/m, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d8 50 05 <unknown>

fmov z0.s, p0/m, #-0.12500000
// CHECK-INST: fmov z0.s, p0/m, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d8 90 05 <unknown>

fmov z0.d, p0/m, #-0.12500000
// CHECK-INST: fmov z0.d, p0/m, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d8 d0 05 <unknown>

fmov z0.d, p0/m, #-0.13281250
// CHECK-INST: fmov z0.d, p0/m, #-0.13281250
// CHECK-ENCODING: [0x20,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d8 d0 05 <unknown>

fmov z0.d, p0/m, #-0.14062500
// CHECK-INST: fmov z0.d, p0/m, #-0.14062500
// CHECK-ENCODING: [0x40,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d8 d0 05 <unknown>

fmov z0.d, p0/m, #-0.14843750
// CHECK-INST: fmov z0.d, p0/m, #-0.14843750
// CHECK-ENCODING: [0x60,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d8 d0 05 <unknown>

fmov z0.d, p0/m, #-0.15625000
// CHECK-INST: fmov z0.d, p0/m, #-0.15625000
// CHECK-ENCODING: [0x80,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d8 d0 05 <unknown>

fmov z0.d, p0/m, #-0.16406250
// CHECK-INST: fmov z0.d, p0/m, #-0.16406250
// CHECK-ENCODING: [0xa0,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d8 d0 05 <unknown>

fmov z0.d, p0/m, #-0.17187500
// CHECK-INST: fmov z0.d, p0/m, #-0.17187500
// CHECK-ENCODING: [0xc0,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d8 d0 05 <unknown>

fmov z0.d, p0/m, #-0.17968750
// CHECK-INST: fmov z0.d, p0/m, #-0.17968750
// CHECK-ENCODING: [0xe0,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d8 d0 05 <unknown>

fmov z0.d, p0/m, #-0.18750000
// CHECK-INST: fmov z0.d, p0/m, #-0.18750000
// CHECK-ENCODING: [0x00,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d9 d0 05 <unknown>

fmov z0.d, p0/m, #-0.19531250
// CHECK-INST: fmov z0.d, p0/m, #-0.19531250
// CHECK-ENCODING: [0x20,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d9 d0 05 <unknown>

fmov z0.d, p0/m, #-0.20312500
// CHECK-INST: fmov z0.d, p0/m, #-0.20312500
// CHECK-ENCODING: [0x40,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d9 d0 05 <unknown>

fmov z0.d, p0/m, #-0.21093750
// CHECK-INST: fmov z0.d, p0/m, #-0.21093750
// CHECK-ENCODING: [0x60,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d9 d0 05 <unknown>

fmov z0.d, p0/m, #-0.21875000
// CHECK-INST: fmov z0.d, p0/m, #-0.21875000
// CHECK-ENCODING: [0x80,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d9 d0 05 <unknown>

fmov z0.d, p0/m, #-0.22656250
// CHECK-INST: fmov z0.d, p0/m, #-0.22656250
// CHECK-ENCODING: [0xa0,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d9 d0 05 <unknown>

fmov z0.d, p0/m, #-0.23437500
// CHECK-INST: fmov z0.d, p0/m, #-0.23437500
// CHECK-ENCODING: [0xc0,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d9 d0 05 <unknown>

fmov z0.d, p0/m, #-0.24218750
// CHECK-INST: fmov z0.d, p0/m, #-0.24218750
// CHECK-ENCODING: [0xe0,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d9 d0 05 <unknown>

fmov z0.d, p0/m, #-0.25000000
// CHECK-INST: fmov z0.d, p0/m, #-0.25000000
// CHECK-ENCODING: [0x00,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 da d0 05 <unknown>

fmov z0.d, p0/m, #-0.26562500
// CHECK-INST: fmov z0.d, p0/m, #-0.26562500
// CHECK-ENCODING: [0x20,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 da d0 05 <unknown>

fmov z0.d, p0/m, #-0.28125000
// CHECK-INST: fmov z0.d, p0/m, #-0.28125000
// CHECK-ENCODING: [0x40,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 da d0 05 <unknown>

fmov z0.d, p0/m, #-0.29687500
// CHECK-INST: fmov z0.d, p0/m, #-0.29687500
// CHECK-ENCODING: [0x60,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 da d0 05 <unknown>

fmov z0.d, p0/m, #-0.31250000
// CHECK-INST: fmov z0.d, p0/m, #-0.31250000
// CHECK-ENCODING: [0x80,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 da d0 05 <unknown>

fmov z0.d, p0/m, #-0.32812500
// CHECK-INST: fmov z0.d, p0/m, #-0.32812500
// CHECK-ENCODING: [0xa0,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 da d0 05 <unknown>

fmov z0.d, p0/m, #-0.34375000
// CHECK-INST: fmov z0.d, p0/m, #-0.34375000
// CHECK-ENCODING: [0xc0,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 da d0 05 <unknown>

fmov z0.d, p0/m, #-0.35937500
// CHECK-INST: fmov z0.d, p0/m, #-0.35937500
// CHECK-ENCODING: [0xe0,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 da d0 05 <unknown>

fmov z0.d, p0/m, #-0.37500000
// CHECK-INST: fmov z0.d, p0/m, #-0.37500000
// CHECK-ENCODING: [0x00,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 db d0 05 <unknown>

fmov z0.d, p0/m, #-0.39062500
// CHECK-INST: fmov z0.d, p0/m, #-0.39062500
// CHECK-ENCODING: [0x20,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 db d0 05 <unknown>

fmov z0.d, p0/m, #-0.40625000
// CHECK-INST: fmov z0.d, p0/m, #-0.40625000
// CHECK-ENCODING: [0x40,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 db d0 05 <unknown>

fmov z0.d, p0/m, #-0.42187500
// CHECK-INST: fmov z0.d, p0/m, #-0.42187500
// CHECK-ENCODING: [0x60,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 db d0 05 <unknown>

fmov z0.d, p0/m, #-0.43750000
// CHECK-INST: fmov z0.d, p0/m, #-0.43750000
// CHECK-ENCODING: [0x80,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 db d0 05 <unknown>

fmov z0.d, p0/m, #-0.45312500
// CHECK-INST: fmov z0.d, p0/m, #-0.45312500
// CHECK-ENCODING: [0xa0,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 db d0 05 <unknown>

fmov z0.d, p0/m, #-0.46875000
// CHECK-INST: fmov z0.d, p0/m, #-0.46875000
// CHECK-ENCODING: [0xc0,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 db d0 05 <unknown>

fmov z0.d, p0/m, #-0.48437500
// CHECK-INST: fmov z0.d, p0/m, #-0.48437500
// CHECK-ENCODING: [0xe0,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 db d0 05 <unknown>

fmov z0.d, p0/m, #-0.50000000
// CHECK-INST: fmov z0.d, p0/m, #-0.50000000
// CHECK-ENCODING: [0x00,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 dc d0 05 <unknown>

fmov z0.d, p0/m, #-0.53125000
// CHECK-INST: fmov z0.d, p0/m, #-0.53125000
// CHECK-ENCODING: [0x20,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 dc d0 05 <unknown>

fmov z0.d, p0/m, #-0.56250000
// CHECK-INST: fmov z0.d, p0/m, #-0.56250000
// CHECK-ENCODING: [0x40,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 dc d0 05 <unknown>

fmov z0.d, p0/m, #-0.59375000
// CHECK-INST: fmov z0.d, p0/m, #-0.59375000
// CHECK-ENCODING: [0x60,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 dc d0 05 <unknown>

fmov z0.d, p0/m, #-0.62500000
// CHECK-INST: fmov z0.d, p0/m, #-0.62500000
// CHECK-ENCODING: [0x80,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 dc d0 05 <unknown>

fmov z0.d, p0/m, #-0.65625000
// CHECK-INST: fmov z0.d, p0/m, #-0.65625000
// CHECK-ENCODING: [0xa0,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 dc d0 05 <unknown>

fmov z0.d, p0/m, #-0.68750000
// CHECK-INST: fmov z0.d, p0/m, #-0.68750000
// CHECK-ENCODING: [0xc0,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 dc d0 05 <unknown>

fmov z0.d, p0/m, #-0.71875000
// CHECK-INST: fmov z0.d, p0/m, #-0.71875000
// CHECK-ENCODING: [0xe0,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 dc d0 05 <unknown>

fmov z0.d, p0/m, #-0.75000000
// CHECK-INST: fmov z0.d, p0/m, #-0.75000000
// CHECK-ENCODING: [0x00,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 dd d0 05 <unknown>

fmov z0.d, p0/m, #-0.78125000
// CHECK-INST: fmov z0.d, p0/m, #-0.78125000
// CHECK-ENCODING: [0x20,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 dd d0 05 <unknown>

fmov z0.d, p0/m, #-0.81250000
// CHECK-INST: fmov z0.d, p0/m, #-0.81250000
// CHECK-ENCODING: [0x40,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 dd d0 05 <unknown>

fmov z0.d, p0/m, #-0.84375000
// CHECK-INST: fmov z0.d, p0/m, #-0.84375000
// CHECK-ENCODING: [0x60,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 dd d0 05 <unknown>

fmov z0.d, p0/m, #-0.87500000
// CHECK-INST: fmov z0.d, p0/m, #-0.87500000
// CHECK-ENCODING: [0x80,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 dd d0 05 <unknown>

fmov z0.d, p0/m, #-0.90625000
// CHECK-INST: fmov z0.d, p0/m, #-0.90625000
// CHECK-ENCODING: [0xa0,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 dd d0 05 <unknown>

fmov z0.d, p0/m, #-0.93750000
// CHECK-INST: fmov z0.d, p0/m, #-0.93750000
// CHECK-ENCODING: [0xc0,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 dd d0 05 <unknown>

fmov z0.d, p0/m, #-0.96875000
// CHECK-INST: fmov z0.d, p0/m, #-0.96875000
// CHECK-ENCODING: [0xe0,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 dd d0 05 <unknown>

fmov z0.d, p0/m, #-1.00000000
// CHECK-INST: fmov z0.d, p0/m, #-1.00000000
// CHECK-ENCODING: [0x00,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 de d0 05 <unknown>

fmov z0.d, p0/m, #-1.06250000
// CHECK-INST: fmov z0.d, p0/m, #-1.06250000
// CHECK-ENCODING: [0x20,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 de d0 05 <unknown>

fmov z0.d, p0/m, #-1.12500000
// CHECK-INST: fmov z0.d, p0/m, #-1.12500000
// CHECK-ENCODING: [0x40,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 de d0 05 <unknown>

fmov z0.d, p0/m, #-1.18750000
// CHECK-INST: fmov z0.d, p0/m, #-1.18750000
// CHECK-ENCODING: [0x60,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 de d0 05 <unknown>

fmov z0.d, p0/m, #-1.25000000
// CHECK-INST: fmov z0.d, p0/m, #-1.25000000
// CHECK-ENCODING: [0x80,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 de d0 05 <unknown>

fmov z0.d, p0/m, #-1.31250000
// CHECK-INST: fmov z0.d, p0/m, #-1.31250000
// CHECK-ENCODING: [0xa0,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 de d0 05 <unknown>

fmov z0.d, p0/m, #-1.37500000
// CHECK-INST: fmov z0.d, p0/m, #-1.37500000
// CHECK-ENCODING: [0xc0,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 de d0 05 <unknown>

fmov z0.d, p0/m, #-1.43750000
// CHECK-INST: fmov z0.d, p0/m, #-1.43750000
// CHECK-ENCODING: [0xe0,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 de d0 05 <unknown>

fmov z0.d, p0/m, #-1.50000000
// CHECK-INST: fmov z0.d, p0/m, #-1.50000000
// CHECK-ENCODING: [0x00,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 df d0 05 <unknown>

fmov z0.d, p0/m, #-1.56250000
// CHECK-INST: fmov z0.d, p0/m, #-1.56250000
// CHECK-ENCODING: [0x20,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 df d0 05 <unknown>

fmov z0.d, p0/m, #-1.62500000
// CHECK-INST: fmov z0.d, p0/m, #-1.62500000
// CHECK-ENCODING: [0x40,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 df d0 05 <unknown>

fmov z0.d, p0/m, #-1.68750000
// CHECK-INST: fmov z0.d, p0/m, #-1.68750000
// CHECK-ENCODING: [0x60,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 df d0 05 <unknown>

fmov z0.d, p0/m, #-1.75000000
// CHECK-INST: fmov z0.d, p0/m, #-1.75000000
// CHECK-ENCODING: [0x80,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 df d0 05 <unknown>

fmov z0.d, p0/m, #-1.81250000
// CHECK-INST: fmov z0.d, p0/m, #-1.81250000
// CHECK-ENCODING: [0xa0,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 df d0 05 <unknown>

fmov z0.d, p0/m, #-1.87500000
// CHECK-INST: fmov z0.d, p0/m, #-1.87500000
// CHECK-ENCODING: [0xc0,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 df d0 05 <unknown>

fmov z0.d, p0/m, #-1.93750000
// CHECK-INST: fmov z0.d, p0/m, #-1.93750000
// CHECK-ENCODING: [0xe0,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 df d0 05 <unknown>

fmov z0.d, p0/m, #-2.00000000
// CHECK-INST: fmov z0.d, p0/m, #-2.00000000
// CHECK-ENCODING: [0x00,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d0 d0 05 <unknown>

fmov z0.d, p0/m, #-2.12500000
// CHECK-INST: fmov z0.d, p0/m, #-2.12500000
// CHECK-ENCODING: [0x20,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d0 d0 05 <unknown>

fmov z0.d, p0/m, #-2.25000000
// CHECK-INST: fmov z0.d, p0/m, #-2.25000000
// CHECK-ENCODING: [0x40,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d0 d0 05 <unknown>

fmov z0.d, p0/m, #-2.37500000
// CHECK-INST: fmov z0.d, p0/m, #-2.37500000
// CHECK-ENCODING: [0x60,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d0 d0 05 <unknown>

fmov z0.d, p0/m, #-2.50000000
// CHECK-INST: fmov z0.d, p0/m, #-2.50000000
// CHECK-ENCODING: [0x80,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d0 d0 05 <unknown>

fmov z0.d, p0/m, #-2.62500000
// CHECK-INST: fmov z0.d, p0/m, #-2.62500000
// CHECK-ENCODING: [0xa0,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d0 d0 05 <unknown>

fmov z0.d, p0/m, #-2.75000000
// CHECK-INST: fmov z0.d, p0/m, #-2.75000000
// CHECK-ENCODING: [0xc0,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d0 d0 05 <unknown>

fmov z0.d, p0/m, #-2.87500000
// CHECK-INST: fmov z0.d, p0/m, #-2.87500000
// CHECK-ENCODING: [0xe0,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d0 d0 05 <unknown>

fmov z0.d, p0/m, #-3.00000000
// CHECK-INST: fmov z0.d, p0/m, #-3.00000000
// CHECK-ENCODING: [0x00,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d1 d0 05 <unknown>

fmov z0.d, p0/m, #-3.12500000
// CHECK-INST: fmov z0.d, p0/m, #-3.12500000
// CHECK-ENCODING: [0x20,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d1 d0 05 <unknown>

fmov z0.d, p0/m, #-3.25000000
// CHECK-INST: fmov z0.d, p0/m, #-3.25000000
// CHECK-ENCODING: [0x40,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d1 d0 05 <unknown>

fmov z0.d, p0/m, #-3.37500000
// CHECK-INST: fmov z0.d, p0/m, #-3.37500000
// CHECK-ENCODING: [0x60,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d1 d0 05 <unknown>

fmov z0.d, p0/m, #-3.50000000
// CHECK-INST: fmov z0.d, p0/m, #-3.50000000
// CHECK-ENCODING: [0x80,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d1 d0 05 <unknown>

fmov z0.d, p0/m, #-3.62500000
// CHECK-INST: fmov z0.d, p0/m, #-3.62500000
// CHECK-ENCODING: [0xa0,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d1 d0 05 <unknown>

fmov z0.d, p0/m, #-3.75000000
// CHECK-INST: fmov z0.d, p0/m, #-3.75000000
// CHECK-ENCODING: [0xc0,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d1 d0 05 <unknown>

fmov z0.d, p0/m, #-3.87500000
// CHECK-INST: fmov z0.d, p0/m, #-3.87500000
// CHECK-ENCODING: [0xe0,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d1 d0 05 <unknown>

fmov z0.d, p0/m, #-4.00000000
// CHECK-INST: fmov z0.d, p0/m, #-4.00000000
// CHECK-ENCODING: [0x00,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d2 d0 05 <unknown>

fmov z0.d, p0/m, #-4.25000000
// CHECK-INST: fmov z0.d, p0/m, #-4.25000000
// CHECK-ENCODING: [0x20,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d2 d0 05 <unknown>

fmov z0.d, p0/m, #-4.50000000
// CHECK-INST: fmov z0.d, p0/m, #-4.50000000
// CHECK-ENCODING: [0x40,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d2 d0 05 <unknown>

fmov z0.d, p0/m, #-4.75000000
// CHECK-INST: fmov z0.d, p0/m, #-4.75000000
// CHECK-ENCODING: [0x60,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d2 d0 05 <unknown>

fmov z0.d, p0/m, #-5.00000000
// CHECK-INST: fmov z0.d, p0/m, #-5.00000000
// CHECK-ENCODING: [0x80,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d2 d0 05 <unknown>

fmov z0.d, p0/m, #-5.25000000
// CHECK-INST: fmov z0.d, p0/m, #-5.25000000
// CHECK-ENCODING: [0xa0,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d2 d0 05 <unknown>

fmov z0.d, p0/m, #-5.50000000
// CHECK-INST: fmov z0.d, p0/m, #-5.50000000
// CHECK-ENCODING: [0xc0,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d2 d0 05 <unknown>

fmov z0.d, p0/m, #-5.75000000
// CHECK-INST: fmov z0.d, p0/m, #-5.75000000
// CHECK-ENCODING: [0xe0,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d2 d0 05 <unknown>

fmov z0.d, p0/m, #-6.00000000
// CHECK-INST: fmov z0.d, p0/m, #-6.00000000
// CHECK-ENCODING: [0x00,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d3 d0 05 <unknown>

fmov z0.d, p0/m, #-6.25000000
// CHECK-INST: fmov z0.d, p0/m, #-6.25000000
// CHECK-ENCODING: [0x20,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d3 d0 05 <unknown>

fmov z0.d, p0/m, #-6.50000000
// CHECK-INST: fmov z0.d, p0/m, #-6.50000000
// CHECK-ENCODING: [0x40,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d3 d0 05 <unknown>

fmov z0.d, p0/m, #-6.75000000
// CHECK-INST: fmov z0.d, p0/m, #-6.75000000
// CHECK-ENCODING: [0x60,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d3 d0 05 <unknown>

fmov z0.d, p0/m, #-7.00000000
// CHECK-INST: fmov z0.d, p0/m, #-7.00000000
// CHECK-ENCODING: [0x80,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d3 d0 05 <unknown>

fmov z0.d, p0/m, #-7.25000000
// CHECK-INST: fmov z0.d, p0/m, #-7.25000000
// CHECK-ENCODING: [0xa0,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d3 d0 05 <unknown>

fmov z0.d, p0/m, #-7.50000000
// CHECK-INST: fmov z0.d, p0/m, #-7.50000000
// CHECK-ENCODING: [0xc0,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d3 d0 05 <unknown>

fmov z0.d, p0/m, #-7.75000000
// CHECK-INST: fmov z0.d, p0/m, #-7.75000000
// CHECK-ENCODING: [0xe0,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d3 d0 05 <unknown>

fmov z0.d, p0/m, #-8.00000000
// CHECK-INST: fmov z0.d, p0/m, #-8.00000000
// CHECK-ENCODING: [0x00,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d4 d0 05 <unknown>

fmov z0.d, p0/m, #-8.50000000
// CHECK-INST: fmov z0.d, p0/m, #-8.50000000
// CHECK-ENCODING: [0x20,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d4 d0 05 <unknown>

fmov z0.d, p0/m, #-9.00000000
// CHECK-INST: fmov z0.d, p0/m, #-9.00000000
// CHECK-ENCODING: [0x40,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d4 d0 05 <unknown>

fmov z0.d, p0/m, #-9.50000000
// CHECK-INST: fmov z0.d, p0/m, #-9.50000000
// CHECK-ENCODING: [0x60,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d4 d0 05 <unknown>

fmov z0.d, p0/m, #-10.00000000
// CHECK-INST: fmov z0.d, p0/m, #-10.00000000
// CHECK-ENCODING: [0x80,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d4 d0 05 <unknown>

fmov z0.d, p0/m, #-10.50000000
// CHECK-INST: fmov z0.d, p0/m, #-10.50000000
// CHECK-ENCODING: [0xa0,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d4 d0 05 <unknown>

fmov z0.d, p0/m, #-11.00000000
// CHECK-INST: fmov z0.d, p0/m, #-11.00000000
// CHECK-ENCODING: [0xc0,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d4 d0 05 <unknown>

fmov z0.d, p0/m, #-11.50000000
// CHECK-INST: fmov z0.d, p0/m, #-11.50000000
// CHECK-ENCODING: [0xe0,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d4 d0 05 <unknown>

fmov z0.d, p0/m, #-12.00000000
// CHECK-INST: fmov z0.d, p0/m, #-12.00000000
// CHECK-ENCODING: [0x00,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d5 d0 05 <unknown>

fmov z0.d, p0/m, #-12.50000000
// CHECK-INST: fmov z0.d, p0/m, #-12.50000000
// CHECK-ENCODING: [0x20,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d5 d0 05 <unknown>

fmov z0.d, p0/m, #-13.00000000
// CHECK-INST: fmov z0.d, p0/m, #-13.00000000
// CHECK-ENCODING: [0x40,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d5 d0 05 <unknown>

fmov z0.d, p0/m, #-13.50000000
// CHECK-INST: fmov z0.d, p0/m, #-13.50000000
// CHECK-ENCODING: [0x60,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d5 d0 05 <unknown>

fmov z0.d, p0/m, #-14.00000000
// CHECK-INST: fmov z0.d, p0/m, #-14.00000000
// CHECK-ENCODING: [0x80,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d5 d0 05 <unknown>

fmov z0.d, p0/m, #-14.50000000
// CHECK-INST: fmov z0.d, p0/m, #-14.50000000
// CHECK-ENCODING: [0xa0,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d5 d0 05 <unknown>

fmov z0.d, p0/m, #-15.00000000
// CHECK-INST: fmov z0.d, p0/m, #-15.00000000
// CHECK-ENCODING: [0xc0,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d5 d0 05 <unknown>

fmov z0.d, p0/m, #-15.50000000
// CHECK-INST: fmov z0.d, p0/m, #-15.50000000
// CHECK-ENCODING: [0xe0,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d5 d0 05 <unknown>

fmov z0.d, p0/m, #-16.00000000
// CHECK-INST: fmov z0.d, p0/m, #-16.00000000
// CHECK-ENCODING: [0x00,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d6 d0 05 <unknown>

fmov z0.d, p0/m, #-17.00000000
// CHECK-INST: fmov z0.d, p0/m, #-17.00000000
// CHECK-ENCODING: [0x20,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d6 d0 05 <unknown>

fmov z0.d, p0/m, #-18.00000000
// CHECK-INST: fmov z0.d, p0/m, #-18.00000000
// CHECK-ENCODING: [0x40,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d6 d0 05 <unknown>

fmov z0.d, p0/m, #-19.00000000
// CHECK-INST: fmov z0.d, p0/m, #-19.00000000
// CHECK-ENCODING: [0x60,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d6 d0 05 <unknown>

fmov z0.d, p0/m, #-20.00000000
// CHECK-INST: fmov z0.d, p0/m, #-20.00000000
// CHECK-ENCODING: [0x80,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d6 d0 05 <unknown>

fmov z0.d, p0/m, #-21.00000000
// CHECK-INST: fmov z0.d, p0/m, #-21.00000000
// CHECK-ENCODING: [0xa0,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d6 d0 05 <unknown>

fmov z0.d, p0/m, #-22.00000000
// CHECK-INST: fmov z0.d, p0/m, #-22.00000000
// CHECK-ENCODING: [0xc0,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d6 d0 05 <unknown>

fmov z0.d, p0/m, #-23.00000000
// CHECK-INST: fmov z0.d, p0/m, #-23.00000000
// CHECK-ENCODING: [0xe0,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d6 d0 05 <unknown>

fmov z0.d, p0/m, #-24.00000000
// CHECK-INST: fmov z0.d, p0/m, #-24.00000000
// CHECK-ENCODING: [0x00,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d7 d0 05 <unknown>

fmov z0.d, p0/m, #-25.00000000
// CHECK-INST: fmov z0.d, p0/m, #-25.00000000
// CHECK-ENCODING: [0x20,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 d7 d0 05 <unknown>

fmov z0.d, p0/m, #-26.00000000
// CHECK-INST: fmov z0.d, p0/m, #-26.00000000
// CHECK-ENCODING: [0x40,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 d7 d0 05 <unknown>

fmov z0.d, p0/m, #-27.00000000
// CHECK-INST: fmov z0.d, p0/m, #-27.00000000
// CHECK-ENCODING: [0x60,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 d7 d0 05 <unknown>

fmov z0.d, p0/m, #-28.00000000
// CHECK-INST: fmov z0.d, p0/m, #-28.00000000
// CHECK-ENCODING: [0x80,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 d7 d0 05 <unknown>

fmov z0.d, p0/m, #-29.00000000
// CHECK-INST: fmov z0.d, p0/m, #-29.00000000
// CHECK-ENCODING: [0xa0,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 d7 d0 05 <unknown>

fmov z0.d, p0/m, #-30.00000000
// CHECK-INST: fmov z0.d, p0/m, #-30.00000000
// CHECK-ENCODING: [0xc0,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 d7 d0 05 <unknown>

fmov z0.d, p0/m, #-31.00000000
// CHECK-INST: fmov z0.d, p0/m, #-31.00000000
// CHECK-ENCODING: [0xe0,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 d7 d0 05 <unknown>

fmov z0.d, p0/m, #0.12500000
// CHECK-INST: fmov z0.d, p0/m, #0.12500000
// CHECK-ENCODING: [0x00,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c8 d0 05 <unknown>

fmov z0.d, p0/m, #0.13281250
// CHECK-INST: fmov z0.d, p0/m, #0.13281250
// CHECK-ENCODING: [0x20,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c8 d0 05 <unknown>

fmov z0.d, p0/m, #0.14062500
// CHECK-INST: fmov z0.d, p0/m, #0.14062500
// CHECK-ENCODING: [0x40,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c8 d0 05 <unknown>

fmov z0.d, p0/m, #0.14843750
// CHECK-INST: fmov z0.d, p0/m, #0.14843750
// CHECK-ENCODING: [0x60,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c8 d0 05 <unknown>

fmov z0.d, p0/m, #0.15625000
// CHECK-INST: fmov z0.d, p0/m, #0.15625000
// CHECK-ENCODING: [0x80,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c8 d0 05 <unknown>

fmov z0.d, p0/m, #0.16406250
// CHECK-INST: fmov z0.d, p0/m, #0.16406250
// CHECK-ENCODING: [0xa0,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c8 d0 05 <unknown>

fmov z0.d, p0/m, #0.17187500
// CHECK-INST: fmov z0.d, p0/m, #0.17187500
// CHECK-ENCODING: [0xc0,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c8 d0 05 <unknown>

fmov z0.d, p0/m, #0.17968750
// CHECK-INST: fmov z0.d, p0/m, #0.17968750
// CHECK-ENCODING: [0xe0,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c8 d0 05 <unknown>

fmov z0.d, p0/m, #0.18750000
// CHECK-INST: fmov z0.d, p0/m, #0.18750000
// CHECK-ENCODING: [0x00,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c9 d0 05 <unknown>

fmov z0.d, p0/m, #0.19531250
// CHECK-INST: fmov z0.d, p0/m, #0.19531250
// CHECK-ENCODING: [0x20,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c9 d0 05 <unknown>

fmov z0.d, p0/m, #0.20312500
// CHECK-INST: fmov z0.d, p0/m, #0.20312500
// CHECK-ENCODING: [0x40,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c9 d0 05 <unknown>

fmov z0.d, p0/m, #0.21093750
// CHECK-INST: fmov z0.d, p0/m, #0.21093750
// CHECK-ENCODING: [0x60,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c9 d0 05 <unknown>

fmov z0.d, p0/m, #0.21875000
// CHECK-INST: fmov z0.d, p0/m, #0.21875000
// CHECK-ENCODING: [0x80,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c9 d0 05 <unknown>

fmov z0.d, p0/m, #0.22656250
// CHECK-INST: fmov z0.d, p0/m, #0.22656250
// CHECK-ENCODING: [0xa0,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c9 d0 05 <unknown>

fmov z0.d, p0/m, #0.23437500
// CHECK-INST: fmov z0.d, p0/m, #0.23437500
// CHECK-ENCODING: [0xc0,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c9 d0 05 <unknown>

fmov z0.d, p0/m, #0.24218750
// CHECK-INST: fmov z0.d, p0/m, #0.24218750
// CHECK-ENCODING: [0xe0,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c9 d0 05 <unknown>

fmov z0.d, p0/m, #0.25000000
// CHECK-INST: fmov z0.d, p0/m, #0.25000000
// CHECK-ENCODING: [0x00,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 ca d0 05 <unknown>

fmov z0.d, p0/m, #0.26562500
// CHECK-INST: fmov z0.d, p0/m, #0.26562500
// CHECK-ENCODING: [0x20,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 ca d0 05 <unknown>

fmov z0.d, p0/m, #0.28125000
// CHECK-INST: fmov z0.d, p0/m, #0.28125000
// CHECK-ENCODING: [0x40,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 ca d0 05 <unknown>

fmov z0.d, p0/m, #0.29687500
// CHECK-INST: fmov z0.d, p0/m, #0.29687500
// CHECK-ENCODING: [0x60,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 ca d0 05 <unknown>

fmov z0.d, p0/m, #0.31250000
// CHECK-INST: fmov z0.d, p0/m, #0.31250000
// CHECK-ENCODING: [0x80,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 ca d0 05 <unknown>

fmov z0.d, p0/m, #0.32812500
// CHECK-INST: fmov z0.d, p0/m, #0.32812500
// CHECK-ENCODING: [0xa0,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 ca d0 05 <unknown>

fmov z0.d, p0/m, #0.34375000
// CHECK-INST: fmov z0.d, p0/m, #0.34375000
// CHECK-ENCODING: [0xc0,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 ca d0 05 <unknown>

fmov z0.d, p0/m, #0.35937500
// CHECK-INST: fmov z0.d, p0/m, #0.35937500
// CHECK-ENCODING: [0xe0,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ca d0 05 <unknown>

fmov z0.d, p0/m, #0.37500000
// CHECK-INST: fmov z0.d, p0/m, #0.37500000
// CHECK-ENCODING: [0x00,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 cb d0 05 <unknown>

fmov z0.d, p0/m, #0.39062500
// CHECK-INST: fmov z0.d, p0/m, #0.39062500
// CHECK-ENCODING: [0x20,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 cb d0 05 <unknown>

fmov z0.d, p0/m, #0.40625000
// CHECK-INST: fmov z0.d, p0/m, #0.40625000
// CHECK-ENCODING: [0x40,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 cb d0 05 <unknown>

fmov z0.d, p0/m, #0.42187500
// CHECK-INST: fmov z0.d, p0/m, #0.42187500
// CHECK-ENCODING: [0x60,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 cb d0 05 <unknown>

fmov z0.d, p0/m, #0.43750000
// CHECK-INST: fmov z0.d, p0/m, #0.43750000
// CHECK-ENCODING: [0x80,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 cb d0 05 <unknown>

fmov z0.d, p0/m, #0.45312500
// CHECK-INST: fmov z0.d, p0/m, #0.45312500
// CHECK-ENCODING: [0xa0,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 cb d0 05 <unknown>

fmov z0.d, p0/m, #0.46875000
// CHECK-INST: fmov z0.d, p0/m, #0.46875000
// CHECK-ENCODING: [0xc0,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 cb d0 05 <unknown>

fmov z0.d, p0/m, #0.48437500
// CHECK-INST: fmov z0.d, p0/m, #0.48437500
// CHECK-ENCODING: [0xe0,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cb d0 05 <unknown>

fmov z0.d, p0/m, #0.50000000
// CHECK-INST: fmov z0.d, p0/m, #0.50000000
// CHECK-ENCODING: [0x00,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 cc d0 05 <unknown>

fmov z0.d, p0/m, #0.53125000
// CHECK-INST: fmov z0.d, p0/m, #0.53125000
// CHECK-ENCODING: [0x20,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 cc d0 05 <unknown>

fmov z0.d, p0/m, #0.56250000
// CHECK-INST: fmov z0.d, p0/m, #0.56250000
// CHECK-ENCODING: [0x40,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 cc d0 05 <unknown>

fmov z0.d, p0/m, #0.59375000
// CHECK-INST: fmov z0.d, p0/m, #0.59375000
// CHECK-ENCODING: [0x60,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 cc d0 05 <unknown>

fmov z0.d, p0/m, #0.62500000
// CHECK-INST: fmov z0.d, p0/m, #0.62500000
// CHECK-ENCODING: [0x80,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 cc d0 05 <unknown>

fmov z0.d, p0/m, #0.65625000
// CHECK-INST: fmov z0.d, p0/m, #0.65625000
// CHECK-ENCODING: [0xa0,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 cc d0 05 <unknown>

fmov z0.d, p0/m, #0.68750000
// CHECK-INST: fmov z0.d, p0/m, #0.68750000
// CHECK-ENCODING: [0xc0,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 cc d0 05 <unknown>

fmov z0.d, p0/m, #0.71875000
// CHECK-INST: fmov z0.d, p0/m, #0.71875000
// CHECK-ENCODING: [0xe0,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cc d0 05 <unknown>

fmov z0.d, p0/m, #0.75000000
// CHECK-INST: fmov z0.d, p0/m, #0.75000000
// CHECK-ENCODING: [0x00,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 cd d0 05 <unknown>

fmov z0.d, p0/m, #0.78125000
// CHECK-INST: fmov z0.d, p0/m, #0.78125000
// CHECK-ENCODING: [0x20,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 cd d0 05 <unknown>

fmov z0.d, p0/m, #0.81250000
// CHECK-INST: fmov z0.d, p0/m, #0.81250000
// CHECK-ENCODING: [0x40,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 cd d0 05 <unknown>

fmov z0.d, p0/m, #0.84375000
// CHECK-INST: fmov z0.d, p0/m, #0.84375000
// CHECK-ENCODING: [0x60,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 cd d0 05 <unknown>

fmov z0.d, p0/m, #0.87500000
// CHECK-INST: fmov z0.d, p0/m, #0.87500000
// CHECK-ENCODING: [0x80,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 cd d0 05 <unknown>

fmov z0.d, p0/m, #0.90625000
// CHECK-INST: fmov z0.d, p0/m, #0.90625000
// CHECK-ENCODING: [0xa0,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 cd d0 05 <unknown>

fmov z0.d, p0/m, #0.93750000
// CHECK-INST: fmov z0.d, p0/m, #0.93750000
// CHECK-ENCODING: [0xc0,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 cd d0 05 <unknown>

fmov z0.d, p0/m, #0.96875000
// CHECK-INST: fmov z0.d, p0/m, #0.96875000
// CHECK-ENCODING: [0xe0,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cd d0 05 <unknown>

fmov z0.d, p0/m, #1.00000000
// CHECK-INST: fmov z0.d, p0/m, #1.00000000
// CHECK-ENCODING: [0x00,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 ce d0 05 <unknown>

fmov z0.d, p0/m, #1.06250000
// CHECK-INST: fmov z0.d, p0/m, #1.06250000
// CHECK-ENCODING: [0x20,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 ce d0 05 <unknown>

fmov z0.d, p0/m, #1.12500000
// CHECK-INST: fmov z0.d, p0/m, #1.12500000
// CHECK-ENCODING: [0x40,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 ce d0 05 <unknown>

fmov z0.d, p0/m, #1.18750000
// CHECK-INST: fmov z0.d, p0/m, #1.18750000
// CHECK-ENCODING: [0x60,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 ce d0 05 <unknown>

fmov z0.d, p0/m, #1.25000000
// CHECK-INST: fmov z0.d, p0/m, #1.25000000
// CHECK-ENCODING: [0x80,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 ce d0 05 <unknown>

fmov z0.d, p0/m, #1.31250000
// CHECK-INST: fmov z0.d, p0/m, #1.31250000
// CHECK-ENCODING: [0xa0,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 ce d0 05 <unknown>

fmov z0.d, p0/m, #1.37500000
// CHECK-INST: fmov z0.d, p0/m, #1.37500000
// CHECK-ENCODING: [0xc0,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 ce d0 05 <unknown>

fmov z0.d, p0/m, #1.43750000
// CHECK-INST: fmov z0.d, p0/m, #1.43750000
// CHECK-ENCODING: [0xe0,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ce d0 05 <unknown>

fmov z0.d, p0/m, #1.50000000
// CHECK-INST: fmov z0.d, p0/m, #1.50000000
// CHECK-ENCODING: [0x00,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 cf d0 05 <unknown>

fmov z0.d, p0/m, #1.56250000
// CHECK-INST: fmov z0.d, p0/m, #1.56250000
// CHECK-ENCODING: [0x20,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 cf d0 05 <unknown>

fmov z0.d, p0/m, #1.62500000
// CHECK-INST: fmov z0.d, p0/m, #1.62500000
// CHECK-ENCODING: [0x40,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 cf d0 05 <unknown>

fmov z0.d, p0/m, #1.68750000
// CHECK-INST: fmov z0.d, p0/m, #1.68750000
// CHECK-ENCODING: [0x60,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 cf d0 05 <unknown>

fmov z0.d, p0/m, #1.75000000
// CHECK-INST: fmov z0.d, p0/m, #1.75000000
// CHECK-ENCODING: [0x80,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 cf d0 05 <unknown>

fmov z0.d, p0/m, #1.81250000
// CHECK-INST: fmov z0.d, p0/m, #1.81250000
// CHECK-ENCODING: [0xa0,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 cf d0 05 <unknown>

fmov z0.d, p0/m, #1.87500000
// CHECK-INST: fmov z0.d, p0/m, #1.87500000
// CHECK-ENCODING: [0xc0,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 cf d0 05 <unknown>

fmov z0.d, p0/m, #1.93750000
// CHECK-INST: fmov z0.d, p0/m, #1.93750000
// CHECK-ENCODING: [0xe0,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cf d0 05 <unknown>

fmov z0.d, p0/m, #2.00000000
// CHECK-INST: fmov z0.d, p0/m, #2.00000000
// CHECK-ENCODING: [0x00,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 d0 05 <unknown>

fmov z0.d, p0/m, #2.12500000
// CHECK-INST: fmov z0.d, p0/m, #2.12500000
// CHECK-ENCODING: [0x20,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c0 d0 05 <unknown>

fmov z0.d, p0/m, #2.25000000
// CHECK-INST: fmov z0.d, p0/m, #2.25000000
// CHECK-ENCODING: [0x40,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c0 d0 05 <unknown>

fmov z0.d, p0/m, #2.37500000
// CHECK-INST: fmov z0.d, p0/m, #2.37500000
// CHECK-ENCODING: [0x60,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c0 d0 05 <unknown>

fmov z0.d, p0/m, #2.50000000
// CHECK-INST: fmov z0.d, p0/m, #2.50000000
// CHECK-ENCODING: [0x80,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c0 d0 05 <unknown>

fmov z0.d, p0/m, #2.62500000
// CHECK-INST: fmov z0.d, p0/m, #2.62500000
// CHECK-ENCODING: [0xa0,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c0 d0 05 <unknown>

fmov z0.d, p0/m, #2.75000000
// CHECK-INST: fmov z0.d, p0/m, #2.75000000
// CHECK-ENCODING: [0xc0,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c0 d0 05 <unknown>

fmov z0.d, p0/m, #2.87500000
// CHECK-INST: fmov z0.d, p0/m, #2.87500000
// CHECK-ENCODING: [0xe0,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c0 d0 05 <unknown>

fmov z0.d, p0/m, #3.00000000
// CHECK-INST: fmov z0.d, p0/m, #3.00000000
// CHECK-ENCODING: [0x00,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c1 d0 05 <unknown>

fmov z0.d, p0/m, #3.12500000
// CHECK-INST: fmov z0.d, p0/m, #3.12500000
// CHECK-ENCODING: [0x20,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c1 d0 05 <unknown>

fmov z0.d, p0/m, #3.25000000
// CHECK-INST: fmov z0.d, p0/m, #3.25000000
// CHECK-ENCODING: [0x40,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c1 d0 05 <unknown>

fmov z0.d, p0/m, #3.37500000
// CHECK-INST: fmov z0.d, p0/m, #3.37500000
// CHECK-ENCODING: [0x60,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c1 d0 05 <unknown>

fmov z0.d, p0/m, #3.50000000
// CHECK-INST: fmov z0.d, p0/m, #3.50000000
// CHECK-ENCODING: [0x80,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c1 d0 05 <unknown>

fmov z0.d, p0/m, #3.62500000
// CHECK-INST: fmov z0.d, p0/m, #3.62500000
// CHECK-ENCODING: [0xa0,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c1 d0 05 <unknown>

fmov z0.d, p0/m, #3.75000000
// CHECK-INST: fmov z0.d, p0/m, #3.75000000
// CHECK-ENCODING: [0xc0,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c1 d0 05 <unknown>

fmov z0.d, p0/m, #3.87500000
// CHECK-INST: fmov z0.d, p0/m, #3.87500000
// CHECK-ENCODING: [0xe0,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c1 d0 05 <unknown>

fmov z0.d, p0/m, #4.00000000
// CHECK-INST: fmov z0.d, p0/m, #4.00000000
// CHECK-ENCODING: [0x00,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c2 d0 05 <unknown>

fmov z0.d, p0/m, #4.25000000
// CHECK-INST: fmov z0.d, p0/m, #4.25000000
// CHECK-ENCODING: [0x20,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c2 d0 05 <unknown>

fmov z0.d, p0/m, #4.50000000
// CHECK-INST: fmov z0.d, p0/m, #4.50000000
// CHECK-ENCODING: [0x40,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c2 d0 05 <unknown>

fmov z0.d, p0/m, #4.75000000
// CHECK-INST: fmov z0.d, p0/m, #4.75000000
// CHECK-ENCODING: [0x60,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c2 d0 05 <unknown>

fmov z0.d, p0/m, #5.00000000
// CHECK-INST: fmov z0.d, p0/m, #5.00000000
// CHECK-ENCODING: [0x80,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c2 d0 05 <unknown>

fmov z0.d, p0/m, #5.25000000
// CHECK-INST: fmov z0.d, p0/m, #5.25000000
// CHECK-ENCODING: [0xa0,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c2 d0 05 <unknown>

fmov z0.d, p0/m, #5.50000000
// CHECK-INST: fmov z0.d, p0/m, #5.50000000
// CHECK-ENCODING: [0xc0,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c2 d0 05 <unknown>

fmov z0.d, p0/m, #5.75000000
// CHECK-INST: fmov z0.d, p0/m, #5.75000000
// CHECK-ENCODING: [0xe0,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c2 d0 05 <unknown>

fmov z0.d, p0/m, #6.00000000
// CHECK-INST: fmov z0.d, p0/m, #6.00000000
// CHECK-ENCODING: [0x00,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c3 d0 05 <unknown>

fmov z0.d, p0/m, #6.25000000
// CHECK-INST: fmov z0.d, p0/m, #6.25000000
// CHECK-ENCODING: [0x20,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c3 d0 05 <unknown>

fmov z0.d, p0/m, #6.50000000
// CHECK-INST: fmov z0.d, p0/m, #6.50000000
// CHECK-ENCODING: [0x40,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c3 d0 05 <unknown>

fmov z0.d, p0/m, #6.75000000
// CHECK-INST: fmov z0.d, p0/m, #6.75000000
// CHECK-ENCODING: [0x60,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c3 d0 05 <unknown>

fmov z0.d, p0/m, #7.00000000
// CHECK-INST: fmov z0.d, p0/m, #7.00000000
// CHECK-ENCODING: [0x80,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c3 d0 05 <unknown>

fmov z0.d, p0/m, #7.25000000
// CHECK-INST: fmov z0.d, p0/m, #7.25000000
// CHECK-ENCODING: [0xa0,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c3 d0 05 <unknown>

fmov z0.d, p0/m, #7.50000000
// CHECK-INST: fmov z0.d, p0/m, #7.50000000
// CHECK-ENCODING: [0xc0,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c3 d0 05 <unknown>

fmov z0.d, p0/m, #7.75000000
// CHECK-INST: fmov z0.d, p0/m, #7.75000000
// CHECK-ENCODING: [0xe0,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c3 d0 05 <unknown>

fmov z0.d, p0/m, #8.00000000
// CHECK-INST: fmov z0.d, p0/m, #8.00000000
// CHECK-ENCODING: [0x00,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c4 d0 05 <unknown>

fmov z0.d, p0/m, #8.50000000
// CHECK-INST: fmov z0.d, p0/m, #8.50000000
// CHECK-ENCODING: [0x20,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c4 d0 05 <unknown>

fmov z0.d, p0/m, #9.00000000
// CHECK-INST: fmov z0.d, p0/m, #9.00000000
// CHECK-ENCODING: [0x40,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c4 d0 05 <unknown>

fmov z0.d, p0/m, #9.50000000
// CHECK-INST: fmov z0.d, p0/m, #9.50000000
// CHECK-ENCODING: [0x60,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c4 d0 05 <unknown>

fmov z0.d, p0/m, #10.00000000
// CHECK-INST: fmov z0.d, p0/m, #10.00000000
// CHECK-ENCODING: [0x80,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c4 d0 05 <unknown>

fmov z0.d, p0/m, #10.50000000
// CHECK-INST: fmov z0.d, p0/m, #10.50000000
// CHECK-ENCODING: [0xa0,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c4 d0 05 <unknown>

fmov z0.d, p0/m, #11.00000000
// CHECK-INST: fmov z0.d, p0/m, #11.00000000
// CHECK-ENCODING: [0xc0,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c4 d0 05 <unknown>

fmov z0.d, p0/m, #11.50000000
// CHECK-INST: fmov z0.d, p0/m, #11.50000000
// CHECK-ENCODING: [0xe0,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c4 d0 05 <unknown>

fmov z0.d, p0/m, #12.00000000
// CHECK-INST: fmov z0.d, p0/m, #12.00000000
// CHECK-ENCODING: [0x00,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c5 d0 05 <unknown>

fmov z0.d, p0/m, #12.50000000
// CHECK-INST: fmov z0.d, p0/m, #12.50000000
// CHECK-ENCODING: [0x20,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c5 d0 05 <unknown>

fmov z0.d, p0/m, #13.00000000
// CHECK-INST: fmov z0.d, p0/m, #13.00000000
// CHECK-ENCODING: [0x40,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c5 d0 05 <unknown>

fmov z0.d, p0/m, #13.50000000
// CHECK-INST: fmov z0.d, p0/m, #13.50000000
// CHECK-ENCODING: [0x60,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c5 d0 05 <unknown>

fmov z0.d, p0/m, #14.00000000
// CHECK-INST: fmov z0.d, p0/m, #14.00000000
// CHECK-ENCODING: [0x80,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c5 d0 05 <unknown>

fmov z0.d, p0/m, #14.50000000
// CHECK-INST: fmov z0.d, p0/m, #14.50000000
// CHECK-ENCODING: [0xa0,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c5 d0 05 <unknown>

fmov z0.d, p0/m, #15.00000000
// CHECK-INST: fmov z0.d, p0/m, #15.00000000
// CHECK-ENCODING: [0xc0,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c5 d0 05 <unknown>

fmov z0.d, p0/m, #15.50000000
// CHECK-INST: fmov z0.d, p0/m, #15.50000000
// CHECK-ENCODING: [0xe0,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c5 d0 05 <unknown>

fmov z0.d, p0/m, #16.00000000
// CHECK-INST: fmov z0.d, p0/m, #16.00000000
// CHECK-ENCODING: [0x00,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c6 d0 05 <unknown>

fmov z0.d, p0/m, #17.00000000
// CHECK-INST: fmov z0.d, p0/m, #17.00000000
// CHECK-ENCODING: [0x20,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c6 d0 05 <unknown>

fmov z0.d, p0/m, #18.00000000
// CHECK-INST: fmov z0.d, p0/m, #18.00000000
// CHECK-ENCODING: [0x40,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c6 d0 05 <unknown>

fmov z0.d, p0/m, #19.00000000
// CHECK-INST: fmov z0.d, p0/m, #19.00000000
// CHECK-ENCODING: [0x60,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c6 d0 05 <unknown>

fmov z0.d, p0/m, #20.00000000
// CHECK-INST: fmov z0.d, p0/m, #20.00000000
// CHECK-ENCODING: [0x80,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c6 d0 05 <unknown>

fmov z0.d, p0/m, #21.00000000
// CHECK-INST: fmov z0.d, p0/m, #21.00000000
// CHECK-ENCODING: [0xa0,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c6 d0 05 <unknown>

fmov z0.d, p0/m, #22.00000000
// CHECK-INST: fmov z0.d, p0/m, #22.00000000
// CHECK-ENCODING: [0xc0,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c6 d0 05 <unknown>

fmov z0.d, p0/m, #23.00000000
// CHECK-INST: fmov z0.d, p0/m, #23.00000000
// CHECK-ENCODING: [0xe0,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c6 d0 05 <unknown>

fmov z0.d, p0/m, #24.00000000
// CHECK-INST: fmov z0.d, p0/m, #24.00000000
// CHECK-ENCODING: [0x00,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c7 d0 05 <unknown>

fmov z0.d, p0/m, #25.00000000
// CHECK-INST: fmov z0.d, p0/m, #25.00000000
// CHECK-ENCODING: [0x20,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 c7 d0 05 <unknown>

fmov z0.d, p0/m, #26.00000000
// CHECK-INST: fmov z0.d, p0/m, #26.00000000
// CHECK-ENCODING: [0x40,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 c7 d0 05 <unknown>

fmov z0.d, p0/m, #27.00000000
// CHECK-INST: fmov z0.d, p0/m, #27.00000000
// CHECK-ENCODING: [0x60,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 c7 d0 05 <unknown>

fmov z0.d, p0/m, #28.00000000
// CHECK-INST: fmov z0.d, p0/m, #28.00000000
// CHECK-ENCODING: [0x80,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 c7 d0 05 <unknown>

fmov z0.d, p0/m, #29.00000000
// CHECK-INST: fmov z0.d, p0/m, #29.00000000
// CHECK-ENCODING: [0xa0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 c7 d0 05 <unknown>

fmov z0.d, p0/m, #30.00000000
// CHECK-INST: fmov z0.d, p0/m, #30.00000000
// CHECK-ENCODING: [0xc0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 c7 d0 05 <unknown>

fmov z0.d, p0/m, #31.00000000
// CHECK-INST: fmov z0.d, p0/m, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c7 d0 05 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0.d, p0/z, z7.d
// CHECK-INST: movprfx	z0.d, p0/z, z7.d
// CHECK-ENCODING: [0xe0,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 20 d0 04 <unknown>

fmov z0.d, p0/m, #31.00000000
// CHECK-INST: fmov	z0.d, p0/m, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c7 d0 05 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

fmov z0.d, p0/m, #31.00000000
// CHECK-INST: fmov	z0.d, p0/m, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c7 d0 05 <unknown>
