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

fdup z0.h, #-0.12500000
// CHECK-INST: fmov z0.h, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0x79,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2579d800 <unknown>

fdup z0.s, #-0.12500000
// CHECK-INST: fmov z0.s, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0xb9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25b9d800 <unknown>

fdup z0.d, #-0.12500000
// CHECK-INST: fmov z0.d, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d800 <unknown>

fdup z0.d, #-0.13281250
// CHECK-INST: fmov z0.d, #-0.13281250
// CHECK-ENCODING: [0x20,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d820 <unknown>

fdup z0.d, #-0.14062500
// CHECK-INST: fmov z0.d, #-0.14062500
// CHECK-ENCODING: [0x40,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d840 <unknown>

fdup z0.d, #-0.14843750
// CHECK-INST: fmov z0.d, #-0.14843750
// CHECK-ENCODING: [0x60,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d860 <unknown>

fdup z0.d, #-0.15625000
// CHECK-INST: fmov z0.d, #-0.15625000
// CHECK-ENCODING: [0x80,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d880 <unknown>

fdup z0.d, #-0.16406250
// CHECK-INST: fmov z0.d, #-0.16406250
// CHECK-ENCODING: [0xa0,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d8a0 <unknown>

fdup z0.d, #-0.17187500
// CHECK-INST: fmov z0.d, #-0.17187500
// CHECK-ENCODING: [0xc0,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d8c0 <unknown>

fdup z0.d, #-0.17968750
// CHECK-INST: fmov z0.d, #-0.17968750
// CHECK-ENCODING: [0xe0,0xd8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d8e0 <unknown>

fdup z0.d, #-0.18750000
// CHECK-INST: fmov z0.d, #-0.18750000
// CHECK-ENCODING: [0x00,0xd9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d900 <unknown>

fdup z0.d, #-0.19531250
// CHECK-INST: fmov z0.d, #-0.19531250
// CHECK-ENCODING: [0x20,0xd9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d920 <unknown>

fdup z0.d, #-0.20312500
// CHECK-INST: fmov z0.d, #-0.20312500
// CHECK-ENCODING: [0x40,0xd9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d940 <unknown>

fdup z0.d, #-0.21093750
// CHECK-INST: fmov z0.d, #-0.21093750
// CHECK-ENCODING: [0x60,0xd9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d960 <unknown>

fdup z0.d, #-0.21875000
// CHECK-INST: fmov z0.d, #-0.21875000
// CHECK-ENCODING: [0x80,0xd9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d980 <unknown>

fdup z0.d, #-0.22656250
// CHECK-INST: fmov z0.d, #-0.22656250
// CHECK-ENCODING: [0xa0,0xd9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d9a0 <unknown>

fdup z0.d, #-0.23437500
// CHECK-INST: fmov z0.d, #-0.23437500
// CHECK-ENCODING: [0xc0,0xd9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d9c0 <unknown>

fdup z0.d, #-0.24218750
// CHECK-INST: fmov z0.d, #-0.24218750
// CHECK-ENCODING: [0xe0,0xd9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d9e0 <unknown>

fdup z0.d, #-0.25000000
// CHECK-INST: fmov z0.d, #-0.25000000
// CHECK-ENCODING: [0x00,0xda,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9da00 <unknown>

fdup z0.d, #-0.26562500
// CHECK-INST: fmov z0.d, #-0.26562500
// CHECK-ENCODING: [0x20,0xda,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9da20 <unknown>

fdup z0.d, #-0.28125000
// CHECK-INST: fmov z0.d, #-0.28125000
// CHECK-ENCODING: [0x40,0xda,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9da40 <unknown>

fdup z0.d, #-0.29687500
// CHECK-INST: fmov z0.d, #-0.29687500
// CHECK-ENCODING: [0x60,0xda,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9da60 <unknown>

fdup z0.d, #-0.31250000
// CHECK-INST: fmov z0.d, #-0.31250000
// CHECK-ENCODING: [0x80,0xda,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9da80 <unknown>

fdup z0.d, #-0.32812500
// CHECK-INST: fmov z0.d, #-0.32812500
// CHECK-ENCODING: [0xa0,0xda,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9daa0 <unknown>

fdup z0.d, #-0.34375000
// CHECK-INST: fmov z0.d, #-0.34375000
// CHECK-ENCODING: [0xc0,0xda,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dac0 <unknown>

fdup z0.d, #-0.35937500
// CHECK-INST: fmov z0.d, #-0.35937500
// CHECK-ENCODING: [0xe0,0xda,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dae0 <unknown>

fdup z0.d, #-0.37500000
// CHECK-INST: fmov z0.d, #-0.37500000
// CHECK-ENCODING: [0x00,0xdb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9db00 <unknown>

fdup z0.d, #-0.39062500
// CHECK-INST: fmov z0.d, #-0.39062500
// CHECK-ENCODING: [0x20,0xdb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9db20 <unknown>

fdup z0.d, #-0.40625000
// CHECK-INST: fmov z0.d, #-0.40625000
// CHECK-ENCODING: [0x40,0xdb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9db40 <unknown>

fdup z0.d, #-0.42187500
// CHECK-INST: fmov z0.d, #-0.42187500
// CHECK-ENCODING: [0x60,0xdb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9db60 <unknown>

fdup z0.d, #-0.43750000
// CHECK-INST: fmov z0.d, #-0.43750000
// CHECK-ENCODING: [0x80,0xdb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9db80 <unknown>

fdup z0.d, #-0.45312500
// CHECK-INST: fmov z0.d, #-0.45312500
// CHECK-ENCODING: [0xa0,0xdb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dba0 <unknown>

fdup z0.d, #-0.46875000
// CHECK-INST: fmov z0.d, #-0.46875000
// CHECK-ENCODING: [0xc0,0xdb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dbc0 <unknown>

fdup z0.d, #-0.48437500
// CHECK-INST: fmov z0.d, #-0.48437500
// CHECK-ENCODING: [0xe0,0xdb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dbe0 <unknown>

fdup z0.d, #-0.50000000
// CHECK-INST: fmov z0.d, #-0.50000000
// CHECK-ENCODING: [0x00,0xdc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dc00 <unknown>

fdup z0.d, #-0.53125000
// CHECK-INST: fmov z0.d, #-0.53125000
// CHECK-ENCODING: [0x20,0xdc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dc20 <unknown>

fdup z0.d, #-0.56250000
// CHECK-INST: fmov z0.d, #-0.56250000
// CHECK-ENCODING: [0x40,0xdc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dc40 <unknown>

fdup z0.d, #-0.59375000
// CHECK-INST: fmov z0.d, #-0.59375000
// CHECK-ENCODING: [0x60,0xdc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dc60 <unknown>

fdup z0.d, #-0.62500000
// CHECK-INST: fmov z0.d, #-0.62500000
// CHECK-ENCODING: [0x80,0xdc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dc80 <unknown>

fdup z0.d, #-0.65625000
// CHECK-INST: fmov z0.d, #-0.65625000
// CHECK-ENCODING: [0xa0,0xdc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dca0 <unknown>

fdup z0.d, #-0.68750000
// CHECK-INST: fmov z0.d, #-0.68750000
// CHECK-ENCODING: [0xc0,0xdc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dcc0 <unknown>

fdup z0.d, #-0.71875000
// CHECK-INST: fmov z0.d, #-0.71875000
// CHECK-ENCODING: [0xe0,0xdc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dce0 <unknown>

fdup z0.d, #-0.75000000
// CHECK-INST: fmov z0.d, #-0.75000000
// CHECK-ENCODING: [0x00,0xdd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dd00 <unknown>

fdup z0.d, #-0.78125000
// CHECK-INST: fmov z0.d, #-0.78125000
// CHECK-ENCODING: [0x20,0xdd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dd20 <unknown>

fdup z0.d, #-0.81250000
// CHECK-INST: fmov z0.d, #-0.81250000
// CHECK-ENCODING: [0x40,0xdd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dd40 <unknown>

fdup z0.d, #-0.84375000
// CHECK-INST: fmov z0.d, #-0.84375000
// CHECK-ENCODING: [0x60,0xdd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dd60 <unknown>

fdup z0.d, #-0.87500000
// CHECK-INST: fmov z0.d, #-0.87500000
// CHECK-ENCODING: [0x80,0xdd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dd80 <unknown>

fdup z0.d, #-0.90625000
// CHECK-INST: fmov z0.d, #-0.90625000
// CHECK-ENCODING: [0xa0,0xdd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dda0 <unknown>

fdup z0.d, #-0.93750000
// CHECK-INST: fmov z0.d, #-0.93750000
// CHECK-ENCODING: [0xc0,0xdd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ddc0 <unknown>

fdup z0.d, #-0.96875000
// CHECK-INST: fmov z0.d, #-0.96875000
// CHECK-ENCODING: [0xe0,0xdd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dde0 <unknown>

fdup z0.d, #-1.00000000
// CHECK-INST: fmov z0.d, #-1.00000000
// CHECK-ENCODING: [0x00,0xde,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9de00 <unknown>

fdup z0.d, #-1.06250000
// CHECK-INST: fmov z0.d, #-1.06250000
// CHECK-ENCODING: [0x20,0xde,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9de20 <unknown>

fdup z0.d, #-1.12500000
// CHECK-INST: fmov z0.d, #-1.12500000
// CHECK-ENCODING: [0x40,0xde,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9de40 <unknown>

fdup z0.d, #-1.18750000
// CHECK-INST: fmov z0.d, #-1.18750000
// CHECK-ENCODING: [0x60,0xde,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9de60 <unknown>

fdup z0.d, #-1.25000000
// CHECK-INST: fmov z0.d, #-1.25000000
// CHECK-ENCODING: [0x80,0xde,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9de80 <unknown>

fdup z0.d, #-1.31250000
// CHECK-INST: fmov z0.d, #-1.31250000
// CHECK-ENCODING: [0xa0,0xde,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dea0 <unknown>

fdup z0.d, #-1.37500000
// CHECK-INST: fmov z0.d, #-1.37500000
// CHECK-ENCODING: [0xc0,0xde,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dec0 <unknown>

fdup z0.d, #-1.43750000
// CHECK-INST: fmov z0.d, #-1.43750000
// CHECK-ENCODING: [0xe0,0xde,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dee0 <unknown>

fdup z0.d, #-1.50000000
// CHECK-INST: fmov z0.d, #-1.50000000
// CHECK-ENCODING: [0x00,0xdf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9df00 <unknown>

fdup z0.d, #-1.56250000
// CHECK-INST: fmov z0.d, #-1.56250000
// CHECK-ENCODING: [0x20,0xdf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9df20 <unknown>

fdup z0.d, #-1.62500000
// CHECK-INST: fmov z0.d, #-1.62500000
// CHECK-ENCODING: [0x40,0xdf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9df40 <unknown>

fdup z0.d, #-1.68750000
// CHECK-INST: fmov z0.d, #-1.68750000
// CHECK-ENCODING: [0x60,0xdf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9df60 <unknown>

fdup z0.d, #-1.75000000
// CHECK-INST: fmov z0.d, #-1.75000000
// CHECK-ENCODING: [0x80,0xdf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9df80 <unknown>

fdup z0.d, #-1.81250000
// CHECK-INST: fmov z0.d, #-1.81250000
// CHECK-ENCODING: [0xa0,0xdf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dfa0 <unknown>

fdup z0.d, #-1.87500000
// CHECK-INST: fmov z0.d, #-1.87500000
// CHECK-ENCODING: [0xc0,0xdf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dfc0 <unknown>

fdup z0.d, #-1.93750000
// CHECK-INST: fmov z0.d, #-1.93750000
// CHECK-ENCODING: [0xe0,0xdf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9dfe0 <unknown>

fdup z0.d, #-2.00000000
// CHECK-INST: fmov z0.d, #-2.00000000
// CHECK-ENCODING: [0x00,0xd0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d000 <unknown>

fdup z0.d, #-2.12500000
// CHECK-INST: fmov z0.d, #-2.12500000
// CHECK-ENCODING: [0x20,0xd0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d020 <unknown>

fdup z0.d, #-2.25000000
// CHECK-INST: fmov z0.d, #-2.25000000
// CHECK-ENCODING: [0x40,0xd0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d040 <unknown>

fdup z0.d, #-2.37500000
// CHECK-INST: fmov z0.d, #-2.37500000
// CHECK-ENCODING: [0x60,0xd0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d060 <unknown>

fdup z0.d, #-2.50000000
// CHECK-INST: fmov z0.d, #-2.50000000
// CHECK-ENCODING: [0x80,0xd0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d080 <unknown>

fdup z0.d, #-2.62500000
// CHECK-INST: fmov z0.d, #-2.62500000
// CHECK-ENCODING: [0xa0,0xd0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d0a0 <unknown>

fdup z0.d, #-2.75000000
// CHECK-INST: fmov z0.d, #-2.75000000
// CHECK-ENCODING: [0xc0,0xd0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d0c0 <unknown>

fdup z0.d, #-2.87500000
// CHECK-INST: fmov z0.d, #-2.87500000
// CHECK-ENCODING: [0xe0,0xd0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d0e0 <unknown>

fdup z0.d, #-3.00000000
// CHECK-INST: fmov z0.d, #-3.00000000
// CHECK-ENCODING: [0x00,0xd1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d100 <unknown>

fdup z0.d, #-3.12500000
// CHECK-INST: fmov z0.d, #-3.12500000
// CHECK-ENCODING: [0x20,0xd1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d120 <unknown>

fdup z0.d, #-3.25000000
// CHECK-INST: fmov z0.d, #-3.25000000
// CHECK-ENCODING: [0x40,0xd1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d140 <unknown>

fdup z0.d, #-3.37500000
// CHECK-INST: fmov z0.d, #-3.37500000
// CHECK-ENCODING: [0x60,0xd1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d160 <unknown>

fdup z0.d, #-3.50000000
// CHECK-INST: fmov z0.d, #-3.50000000
// CHECK-ENCODING: [0x80,0xd1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d180 <unknown>

fdup z0.d, #-3.62500000
// CHECK-INST: fmov z0.d, #-3.62500000
// CHECK-ENCODING: [0xa0,0xd1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d1a0 <unknown>

fdup z0.d, #-3.75000000
// CHECK-INST: fmov z0.d, #-3.75000000
// CHECK-ENCODING: [0xc0,0xd1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d1c0 <unknown>

fdup z0.d, #-3.87500000
// CHECK-INST: fmov z0.d, #-3.87500000
// CHECK-ENCODING: [0xe0,0xd1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d1e0 <unknown>

fdup z0.d, #-4.00000000
// CHECK-INST: fmov z0.d, #-4.00000000
// CHECK-ENCODING: [0x00,0xd2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d200 <unknown>

fdup z0.d, #-4.25000000
// CHECK-INST: fmov z0.d, #-4.25000000
// CHECK-ENCODING: [0x20,0xd2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d220 <unknown>

fdup z0.d, #-4.50000000
// CHECK-INST: fmov z0.d, #-4.50000000
// CHECK-ENCODING: [0x40,0xd2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d240 <unknown>

fdup z0.d, #-4.75000000
// CHECK-INST: fmov z0.d, #-4.75000000
// CHECK-ENCODING: [0x60,0xd2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d260 <unknown>

fdup z0.d, #-5.00000000
// CHECK-INST: fmov z0.d, #-5.00000000
// CHECK-ENCODING: [0x80,0xd2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d280 <unknown>

fdup z0.d, #-5.25000000
// CHECK-INST: fmov z0.d, #-5.25000000
// CHECK-ENCODING: [0xa0,0xd2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d2a0 <unknown>

fdup z0.d, #-5.50000000
// CHECK-INST: fmov z0.d, #-5.50000000
// CHECK-ENCODING: [0xc0,0xd2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d2c0 <unknown>

fdup z0.d, #-5.75000000
// CHECK-INST: fmov z0.d, #-5.75000000
// CHECK-ENCODING: [0xe0,0xd2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d2e0 <unknown>

fdup z0.d, #-6.00000000
// CHECK-INST: fmov z0.d, #-6.00000000
// CHECK-ENCODING: [0x00,0xd3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d300 <unknown>

fdup z0.d, #-6.25000000
// CHECK-INST: fmov z0.d, #-6.25000000
// CHECK-ENCODING: [0x20,0xd3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d320 <unknown>

fdup z0.d, #-6.50000000
// CHECK-INST: fmov z0.d, #-6.50000000
// CHECK-ENCODING: [0x40,0xd3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d340 <unknown>

fdup z0.d, #-6.75000000
// CHECK-INST: fmov z0.d, #-6.75000000
// CHECK-ENCODING: [0x60,0xd3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d360 <unknown>

fdup z0.d, #-7.00000000
// CHECK-INST: fmov z0.d, #-7.00000000
// CHECK-ENCODING: [0x80,0xd3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d380 <unknown>

fdup z0.d, #-7.25000000
// CHECK-INST: fmov z0.d, #-7.25000000
// CHECK-ENCODING: [0xa0,0xd3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d3a0 <unknown>

fdup z0.d, #-7.50000000
// CHECK-INST: fmov z0.d, #-7.50000000
// CHECK-ENCODING: [0xc0,0xd3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d3c0 <unknown>

fdup z0.d, #-7.75000000
// CHECK-INST: fmov z0.d, #-7.75000000
// CHECK-ENCODING: [0xe0,0xd3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d3e0 <unknown>

fdup z0.d, #-8.00000000
// CHECK-INST: fmov z0.d, #-8.00000000
// CHECK-ENCODING: [0x00,0xd4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d400 <unknown>

fdup z0.d, #-8.50000000
// CHECK-INST: fmov z0.d, #-8.50000000
// CHECK-ENCODING: [0x20,0xd4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d420 <unknown>

fdup z0.d, #-9.00000000
// CHECK-INST: fmov z0.d, #-9.00000000
// CHECK-ENCODING: [0x40,0xd4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d440 <unknown>

fdup z0.d, #-9.50000000
// CHECK-INST: fmov z0.d, #-9.50000000
// CHECK-ENCODING: [0x60,0xd4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d460 <unknown>

fdup z0.d, #-10.00000000
// CHECK-INST: fmov z0.d, #-10.00000000
// CHECK-ENCODING: [0x80,0xd4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d480 <unknown>

fdup z0.d, #-10.50000000
// CHECK-INST: fmov z0.d, #-10.50000000
// CHECK-ENCODING: [0xa0,0xd4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d4a0 <unknown>

fdup z0.d, #-11.00000000
// CHECK-INST: fmov z0.d, #-11.00000000
// CHECK-ENCODING: [0xc0,0xd4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d4c0 <unknown>

fdup z0.d, #-11.50000000
// CHECK-INST: fmov z0.d, #-11.50000000
// CHECK-ENCODING: [0xe0,0xd4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d4e0 <unknown>

fdup z0.d, #-12.00000000
// CHECK-INST: fmov z0.d, #-12.00000000
// CHECK-ENCODING: [0x00,0xd5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d500 <unknown>

fdup z0.d, #-12.50000000
// CHECK-INST: fmov z0.d, #-12.50000000
// CHECK-ENCODING: [0x20,0xd5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d520 <unknown>

fdup z0.d, #-13.00000000
// CHECK-INST: fmov z0.d, #-13.00000000
// CHECK-ENCODING: [0x40,0xd5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d540 <unknown>

fdup z0.d, #-13.50000000
// CHECK-INST: fmov z0.d, #-13.50000000
// CHECK-ENCODING: [0x60,0xd5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d560 <unknown>

fdup z0.d, #-14.00000000
// CHECK-INST: fmov z0.d, #-14.00000000
// CHECK-ENCODING: [0x80,0xd5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d580 <unknown>

fdup z0.d, #-14.50000000
// CHECK-INST: fmov z0.d, #-14.50000000
// CHECK-ENCODING: [0xa0,0xd5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d5a0 <unknown>

fdup z0.d, #-15.00000000
// CHECK-INST: fmov z0.d, #-15.00000000
// CHECK-ENCODING: [0xc0,0xd5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d5c0 <unknown>

fdup z0.d, #-15.50000000
// CHECK-INST: fmov z0.d, #-15.50000000
// CHECK-ENCODING: [0xe0,0xd5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d5e0 <unknown>

fdup z0.d, #-16.00000000
// CHECK-INST: fmov z0.d, #-16.00000000
// CHECK-ENCODING: [0x00,0xd6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d600 <unknown>

fdup z0.d, #-17.00000000
// CHECK-INST: fmov z0.d, #-17.00000000
// CHECK-ENCODING: [0x20,0xd6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d620 <unknown>

fdup z0.d, #-18.00000000
// CHECK-INST: fmov z0.d, #-18.00000000
// CHECK-ENCODING: [0x40,0xd6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d640 <unknown>

fdup z0.d, #-19.00000000
// CHECK-INST: fmov z0.d, #-19.00000000
// CHECK-ENCODING: [0x60,0xd6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d660 <unknown>

fdup z0.d, #-20.00000000
// CHECK-INST: fmov z0.d, #-20.00000000
// CHECK-ENCODING: [0x80,0xd6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d680 <unknown>

fdup z0.d, #-21.00000000
// CHECK-INST: fmov z0.d, #-21.00000000
// CHECK-ENCODING: [0xa0,0xd6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d6a0 <unknown>

fdup z0.d, #-22.00000000
// CHECK-INST: fmov z0.d, #-22.00000000
// CHECK-ENCODING: [0xc0,0xd6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d6c0 <unknown>

fdup z0.d, #-23.00000000
// CHECK-INST: fmov z0.d, #-23.00000000
// CHECK-ENCODING: [0xe0,0xd6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d6e0 <unknown>

fdup z0.d, #-24.00000000
// CHECK-INST: fmov z0.d, #-24.00000000
// CHECK-ENCODING: [0x00,0xd7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d700 <unknown>

fdup z0.d, #-25.00000000
// CHECK-INST: fmov z0.d, #-25.00000000
// CHECK-ENCODING: [0x20,0xd7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d720 <unknown>

fdup z0.d, #-26.00000000
// CHECK-INST: fmov z0.d, #-26.00000000
// CHECK-ENCODING: [0x40,0xd7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d740 <unknown>

fdup z0.d, #-27.00000000
// CHECK-INST: fmov z0.d, #-27.00000000
// CHECK-ENCODING: [0x60,0xd7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d760 <unknown>

fdup z0.d, #-28.00000000
// CHECK-INST: fmov z0.d, #-28.00000000
// CHECK-ENCODING: [0x80,0xd7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d780 <unknown>

fdup z0.d, #-29.00000000
// CHECK-INST: fmov z0.d, #-29.00000000
// CHECK-ENCODING: [0xa0,0xd7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d7a0 <unknown>

fdup z0.d, #-30.00000000
// CHECK-INST: fmov z0.d, #-30.00000000
// CHECK-ENCODING: [0xc0,0xd7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d7c0 <unknown>

fdup z0.d, #-31.00000000
// CHECK-INST: fmov z0.d, #-31.00000000
// CHECK-ENCODING: [0xe0,0xd7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9d7e0 <unknown>

fdup z0.d, #0.12500000
// CHECK-INST: fmov z0.d, #0.12500000
// CHECK-ENCODING: [0x00,0xc8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c800 <unknown>

fdup z0.d, #0.13281250
// CHECK-INST: fmov z0.d, #0.13281250
// CHECK-ENCODING: [0x20,0xc8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c820 <unknown>

fdup z0.d, #0.14062500
// CHECK-INST: fmov z0.d, #0.14062500
// CHECK-ENCODING: [0x40,0xc8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c840 <unknown>

fdup z0.d, #0.14843750
// CHECK-INST: fmov z0.d, #0.14843750
// CHECK-ENCODING: [0x60,0xc8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c860 <unknown>

fdup z0.d, #0.15625000
// CHECK-INST: fmov z0.d, #0.15625000
// CHECK-ENCODING: [0x80,0xc8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c880 <unknown>

fdup z0.d, #0.16406250
// CHECK-INST: fmov z0.d, #0.16406250
// CHECK-ENCODING: [0xa0,0xc8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c8a0 <unknown>

fdup z0.d, #0.17187500
// CHECK-INST: fmov z0.d, #0.17187500
// CHECK-ENCODING: [0xc0,0xc8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c8c0 <unknown>

fdup z0.d, #0.17968750
// CHECK-INST: fmov z0.d, #0.17968750
// CHECK-ENCODING: [0xe0,0xc8,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c8e0 <unknown>

fdup z0.d, #0.18750000
// CHECK-INST: fmov z0.d, #0.18750000
// CHECK-ENCODING: [0x00,0xc9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c900 <unknown>

fdup z0.d, #0.19531250
// CHECK-INST: fmov z0.d, #0.19531250
// CHECK-ENCODING: [0x20,0xc9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c920 <unknown>

fdup z0.d, #0.20312500
// CHECK-INST: fmov z0.d, #0.20312500
// CHECK-ENCODING: [0x40,0xc9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c940 <unknown>

fdup z0.d, #0.21093750
// CHECK-INST: fmov z0.d, #0.21093750
// CHECK-ENCODING: [0x60,0xc9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c960 <unknown>

fdup z0.d, #0.21875000
// CHECK-INST: fmov z0.d, #0.21875000
// CHECK-ENCODING: [0x80,0xc9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c980 <unknown>

fdup z0.d, #0.22656250
// CHECK-INST: fmov z0.d, #0.22656250
// CHECK-ENCODING: [0xa0,0xc9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c9a0 <unknown>

fdup z0.d, #0.23437500
// CHECK-INST: fmov z0.d, #0.23437500
// CHECK-ENCODING: [0xc0,0xc9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c9c0 <unknown>

fdup z0.d, #0.24218750
// CHECK-INST: fmov z0.d, #0.24218750
// CHECK-ENCODING: [0xe0,0xc9,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c9e0 <unknown>

fdup z0.d, #0.25000000
// CHECK-INST: fmov z0.d, #0.25000000
// CHECK-ENCODING: [0x00,0xca,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ca00 <unknown>

fdup z0.d, #0.26562500
// CHECK-INST: fmov z0.d, #0.26562500
// CHECK-ENCODING: [0x20,0xca,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ca20 <unknown>

fdup z0.d, #0.28125000
// CHECK-INST: fmov z0.d, #0.28125000
// CHECK-ENCODING: [0x40,0xca,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ca40 <unknown>

fdup z0.d, #0.29687500
// CHECK-INST: fmov z0.d, #0.29687500
// CHECK-ENCODING: [0x60,0xca,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ca60 <unknown>

fdup z0.d, #0.31250000
// CHECK-INST: fmov z0.d, #0.31250000
// CHECK-ENCODING: [0x80,0xca,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ca80 <unknown>

fdup z0.d, #0.32812500
// CHECK-INST: fmov z0.d, #0.32812500
// CHECK-ENCODING: [0xa0,0xca,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9caa0 <unknown>

fdup z0.d, #0.34375000
// CHECK-INST: fmov z0.d, #0.34375000
// CHECK-ENCODING: [0xc0,0xca,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cac0 <unknown>

fdup z0.d, #0.35937500
// CHECK-INST: fmov z0.d, #0.35937500
// CHECK-ENCODING: [0xe0,0xca,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cae0 <unknown>

fdup z0.d, #0.37500000
// CHECK-INST: fmov z0.d, #0.37500000
// CHECK-ENCODING: [0x00,0xcb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cb00 <unknown>

fdup z0.d, #0.39062500
// CHECK-INST: fmov z0.d, #0.39062500
// CHECK-ENCODING: [0x20,0xcb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cb20 <unknown>

fdup z0.d, #0.40625000
// CHECK-INST: fmov z0.d, #0.40625000
// CHECK-ENCODING: [0x40,0xcb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cb40 <unknown>

fdup z0.d, #0.42187500
// CHECK-INST: fmov z0.d, #0.42187500
// CHECK-ENCODING: [0x60,0xcb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cb60 <unknown>

fdup z0.d, #0.43750000
// CHECK-INST: fmov z0.d, #0.43750000
// CHECK-ENCODING: [0x80,0xcb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cb80 <unknown>

fdup z0.d, #0.45312500
// CHECK-INST: fmov z0.d, #0.45312500
// CHECK-ENCODING: [0xa0,0xcb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cba0 <unknown>

fdup z0.d, #0.46875000
// CHECK-INST: fmov z0.d, #0.46875000
// CHECK-ENCODING: [0xc0,0xcb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cbc0 <unknown>

fdup z0.d, #0.48437500
// CHECK-INST: fmov z0.d, #0.48437500
// CHECK-ENCODING: [0xe0,0xcb,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cbe0 <unknown>

fdup z0.d, #0.50000000
// CHECK-INST: fmov z0.d, #0.50000000
// CHECK-ENCODING: [0x00,0xcc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cc00 <unknown>

fdup z0.d, #0.53125000
// CHECK-INST: fmov z0.d, #0.53125000
// CHECK-ENCODING: [0x20,0xcc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cc20 <unknown>

fdup z0.d, #0.56250000
// CHECK-INST: fmov z0.d, #0.56250000
// CHECK-ENCODING: [0x40,0xcc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cc40 <unknown>

fdup z0.d, #0.59375000
// CHECK-INST: fmov z0.d, #0.59375000
// CHECK-ENCODING: [0x60,0xcc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cc60 <unknown>

fdup z0.d, #0.62500000
// CHECK-INST: fmov z0.d, #0.62500000
// CHECK-ENCODING: [0x80,0xcc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cc80 <unknown>

fdup z0.d, #0.65625000
// CHECK-INST: fmov z0.d, #0.65625000
// CHECK-ENCODING: [0xa0,0xcc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cca0 <unknown>

fdup z0.d, #0.68750000
// CHECK-INST: fmov z0.d, #0.68750000
// CHECK-ENCODING: [0xc0,0xcc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ccc0 <unknown>

fdup z0.d, #0.71875000
// CHECK-INST: fmov z0.d, #0.71875000
// CHECK-ENCODING: [0xe0,0xcc,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cce0 <unknown>

fdup z0.d, #0.75000000
// CHECK-INST: fmov z0.d, #0.75000000
// CHECK-ENCODING: [0x00,0xcd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cd00 <unknown>

fdup z0.d, #0.78125000
// CHECK-INST: fmov z0.d, #0.78125000
// CHECK-ENCODING: [0x20,0xcd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cd20 <unknown>

fdup z0.d, #0.81250000
// CHECK-INST: fmov z0.d, #0.81250000
// CHECK-ENCODING: [0x40,0xcd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cd40 <unknown>

fdup z0.d, #0.84375000
// CHECK-INST: fmov z0.d, #0.84375000
// CHECK-ENCODING: [0x60,0xcd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cd60 <unknown>

fdup z0.d, #0.87500000
// CHECK-INST: fmov z0.d, #0.87500000
// CHECK-ENCODING: [0x80,0xcd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cd80 <unknown>

fdup z0.d, #0.90625000
// CHECK-INST: fmov z0.d, #0.90625000
// CHECK-ENCODING: [0xa0,0xcd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cda0 <unknown>

fdup z0.d, #0.93750000
// CHECK-INST: fmov z0.d, #0.93750000
// CHECK-ENCODING: [0xc0,0xcd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cdc0 <unknown>

fdup z0.d, #0.96875000
// CHECK-INST: fmov z0.d, #0.96875000
// CHECK-ENCODING: [0xe0,0xcd,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cde0 <unknown>

fdup z0.d, #1.00000000
// CHECK-INST: fmov z0.d, #1.00000000
// CHECK-ENCODING: [0x00,0xce,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ce00 <unknown>

fdup z0.d, #1.06250000
// CHECK-INST: fmov z0.d, #1.06250000
// CHECK-ENCODING: [0x20,0xce,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ce20 <unknown>

fdup z0.d, #1.12500000
// CHECK-INST: fmov z0.d, #1.12500000
// CHECK-ENCODING: [0x40,0xce,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ce40 <unknown>

fdup z0.d, #1.18750000
// CHECK-INST: fmov z0.d, #1.18750000
// CHECK-ENCODING: [0x60,0xce,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ce60 <unknown>

fdup z0.d, #1.25000000
// CHECK-INST: fmov z0.d, #1.25000000
// CHECK-ENCODING: [0x80,0xce,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9ce80 <unknown>

fdup z0.d, #1.31250000
// CHECK-INST: fmov z0.d, #1.31250000
// CHECK-ENCODING: [0xa0,0xce,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cea0 <unknown>

fdup z0.d, #1.37500000
// CHECK-INST: fmov z0.d, #1.37500000
// CHECK-ENCODING: [0xc0,0xce,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cec0 <unknown>

fdup z0.d, #1.43750000
// CHECK-INST: fmov z0.d, #1.43750000
// CHECK-ENCODING: [0xe0,0xce,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cee0 <unknown>

fdup z0.d, #1.50000000
// CHECK-INST: fmov z0.d, #1.50000000
// CHECK-ENCODING: [0x00,0xcf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cf00 <unknown>

fdup z0.d, #1.56250000
// CHECK-INST: fmov z0.d, #1.56250000
// CHECK-ENCODING: [0x20,0xcf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cf20 <unknown>

fdup z0.d, #1.62500000
// CHECK-INST: fmov z0.d, #1.62500000
// CHECK-ENCODING: [0x40,0xcf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cf40 <unknown>

fdup z0.d, #1.68750000
// CHECK-INST: fmov z0.d, #1.68750000
// CHECK-ENCODING: [0x60,0xcf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cf60 <unknown>

fdup z0.d, #1.75000000
// CHECK-INST: fmov z0.d, #1.75000000
// CHECK-ENCODING: [0x80,0xcf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cf80 <unknown>

fdup z0.d, #1.81250000
// CHECK-INST: fmov z0.d, #1.81250000
// CHECK-ENCODING: [0xa0,0xcf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cfa0 <unknown>

fdup z0.d, #1.87500000
// CHECK-INST: fmov z0.d, #1.87500000
// CHECK-ENCODING: [0xc0,0xcf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cfc0 <unknown>

fdup z0.d, #1.93750000
// CHECK-INST: fmov z0.d, #1.93750000
// CHECK-ENCODING: [0xe0,0xcf,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9cfe0 <unknown>

fdup z0.d, #2.00000000
// CHECK-INST: fmov z0.d, #2.00000000
// CHECK-ENCODING: [0x00,0xc0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c000 <unknown>

fdup z0.d, #2.12500000
// CHECK-INST: fmov z0.d, #2.12500000
// CHECK-ENCODING: [0x20,0xc0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c020 <unknown>

fdup z0.d, #2.25000000
// CHECK-INST: fmov z0.d, #2.25000000
// CHECK-ENCODING: [0x40,0xc0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c040 <unknown>

fdup z0.d, #2.37500000
// CHECK-INST: fmov z0.d, #2.37500000
// CHECK-ENCODING: [0x60,0xc0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c060 <unknown>

fdup z0.d, #2.50000000
// CHECK-INST: fmov z0.d, #2.50000000
// CHECK-ENCODING: [0x80,0xc0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c080 <unknown>

fdup z0.d, #2.62500000
// CHECK-INST: fmov z0.d, #2.62500000
// CHECK-ENCODING: [0xa0,0xc0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c0a0 <unknown>

fdup z0.d, #2.75000000
// CHECK-INST: fmov z0.d, #2.75000000
// CHECK-ENCODING: [0xc0,0xc0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c0c0 <unknown>

fdup z0.d, #2.87500000
// CHECK-INST: fmov z0.d, #2.87500000
// CHECK-ENCODING: [0xe0,0xc0,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c0e0 <unknown>

fdup z0.d, #3.00000000
// CHECK-INST: fmov z0.d, #3.00000000
// CHECK-ENCODING: [0x00,0xc1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c100 <unknown>

fdup z0.d, #3.12500000
// CHECK-INST: fmov z0.d, #3.12500000
// CHECK-ENCODING: [0x20,0xc1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c120 <unknown>

fdup z0.d, #3.25000000
// CHECK-INST: fmov z0.d, #3.25000000
// CHECK-ENCODING: [0x40,0xc1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c140 <unknown>

fdup z0.d, #3.37500000
// CHECK-INST: fmov z0.d, #3.37500000
// CHECK-ENCODING: [0x60,0xc1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c160 <unknown>

fdup z0.d, #3.50000000
// CHECK-INST: fmov z0.d, #3.50000000
// CHECK-ENCODING: [0x80,0xc1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c180 <unknown>

fdup z0.d, #3.62500000
// CHECK-INST: fmov z0.d, #3.62500000
// CHECK-ENCODING: [0xa0,0xc1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c1a0 <unknown>

fdup z0.d, #3.75000000
// CHECK-INST: fmov z0.d, #3.75000000
// CHECK-ENCODING: [0xc0,0xc1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c1c0 <unknown>

fdup z0.d, #3.87500000
// CHECK-INST: fmov z0.d, #3.87500000
// CHECK-ENCODING: [0xe0,0xc1,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c1e0 <unknown>

fdup z0.d, #4.00000000
// CHECK-INST: fmov z0.d, #4.00000000
// CHECK-ENCODING: [0x00,0xc2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c200 <unknown>

fdup z0.d, #4.25000000
// CHECK-INST: fmov z0.d, #4.25000000
// CHECK-ENCODING: [0x20,0xc2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c220 <unknown>

fdup z0.d, #4.50000000
// CHECK-INST: fmov z0.d, #4.50000000
// CHECK-ENCODING: [0x40,0xc2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c240 <unknown>

fdup z0.d, #4.75000000
// CHECK-INST: fmov z0.d, #4.75000000
// CHECK-ENCODING: [0x60,0xc2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c260 <unknown>

fdup z0.d, #5.00000000
// CHECK-INST: fmov z0.d, #5.00000000
// CHECK-ENCODING: [0x80,0xc2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c280 <unknown>

fdup z0.d, #5.25000000
// CHECK-INST: fmov z0.d, #5.25000000
// CHECK-ENCODING: [0xa0,0xc2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c2a0 <unknown>

fdup z0.d, #5.50000000
// CHECK-INST: fmov z0.d, #5.50000000
// CHECK-ENCODING: [0xc0,0xc2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c2c0 <unknown>

fdup z0.d, #5.75000000
// CHECK-INST: fmov z0.d, #5.75000000
// CHECK-ENCODING: [0xe0,0xc2,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c2e0 <unknown>

fdup z0.d, #6.00000000
// CHECK-INST: fmov z0.d, #6.00000000
// CHECK-ENCODING: [0x00,0xc3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c300 <unknown>

fdup z0.d, #6.25000000
// CHECK-INST: fmov z0.d, #6.25000000
// CHECK-ENCODING: [0x20,0xc3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c320 <unknown>

fdup z0.d, #6.50000000
// CHECK-INST: fmov z0.d, #6.50000000
// CHECK-ENCODING: [0x40,0xc3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c340 <unknown>

fdup z0.d, #6.75000000
// CHECK-INST: fmov z0.d, #6.75000000
// CHECK-ENCODING: [0x60,0xc3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c360 <unknown>

fdup z0.d, #7.00000000
// CHECK-INST: fmov z0.d, #7.00000000
// CHECK-ENCODING: [0x80,0xc3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c380 <unknown>

fdup z0.d, #7.25000000
// CHECK-INST: fmov z0.d, #7.25000000
// CHECK-ENCODING: [0xa0,0xc3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c3a0 <unknown>

fdup z0.d, #7.50000000
// CHECK-INST: fmov z0.d, #7.50000000
// CHECK-ENCODING: [0xc0,0xc3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c3c0 <unknown>

fdup z0.d, #7.75000000
// CHECK-INST: fmov z0.d, #7.75000000
// CHECK-ENCODING: [0xe0,0xc3,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c3e0 <unknown>

fdup z0.d, #8.00000000
// CHECK-INST: fmov z0.d, #8.00000000
// CHECK-ENCODING: [0x00,0xc4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c400 <unknown>

fdup z0.d, #8.50000000
// CHECK-INST: fmov z0.d, #8.50000000
// CHECK-ENCODING: [0x20,0xc4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c420 <unknown>

fdup z0.d, #9.00000000
// CHECK-INST: fmov z0.d, #9.00000000
// CHECK-ENCODING: [0x40,0xc4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c440 <unknown>

fdup z0.d, #9.50000000
// CHECK-INST: fmov z0.d, #9.50000000
// CHECK-ENCODING: [0x60,0xc4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c460 <unknown>

fdup z0.d, #10.00000000
// CHECK-INST: fmov z0.d, #10.00000000
// CHECK-ENCODING: [0x80,0xc4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c480 <unknown>

fdup z0.d, #10.50000000
// CHECK-INST: fmov z0.d, #10.50000000
// CHECK-ENCODING: [0xa0,0xc4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c4a0 <unknown>

fdup z0.d, #11.00000000
// CHECK-INST: fmov z0.d, #11.00000000
// CHECK-ENCODING: [0xc0,0xc4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c4c0 <unknown>

fdup z0.d, #11.50000000
// CHECK-INST: fmov z0.d, #11.50000000
// CHECK-ENCODING: [0xe0,0xc4,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c4e0 <unknown>

fdup z0.d, #12.00000000
// CHECK-INST: fmov z0.d, #12.00000000
// CHECK-ENCODING: [0x00,0xc5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c500 <unknown>

fdup z0.d, #12.50000000
// CHECK-INST: fmov z0.d, #12.50000000
// CHECK-ENCODING: [0x20,0xc5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c520 <unknown>

fdup z0.d, #13.00000000
// CHECK-INST: fmov z0.d, #13.00000000
// CHECK-ENCODING: [0x40,0xc5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c540 <unknown>

fdup z0.d, #13.50000000
// CHECK-INST: fmov z0.d, #13.50000000
// CHECK-ENCODING: [0x60,0xc5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c560 <unknown>

fdup z0.d, #14.00000000
// CHECK-INST: fmov z0.d, #14.00000000
// CHECK-ENCODING: [0x80,0xc5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c580 <unknown>

fdup z0.d, #14.50000000
// CHECK-INST: fmov z0.d, #14.50000000
// CHECK-ENCODING: [0xa0,0xc5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c5a0 <unknown>

fdup z0.d, #15.00000000
// CHECK-INST: fmov z0.d, #15.00000000
// CHECK-ENCODING: [0xc0,0xc5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c5c0 <unknown>

fdup z0.d, #15.50000000
// CHECK-INST: fmov z0.d, #15.50000000
// CHECK-ENCODING: [0xe0,0xc5,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c5e0 <unknown>

fdup z0.d, #16.00000000
// CHECK-INST: fmov z0.d, #16.00000000
// CHECK-ENCODING: [0x00,0xc6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c600 <unknown>

fdup z0.d, #17.00000000
// CHECK-INST: fmov z0.d, #17.00000000
// CHECK-ENCODING: [0x20,0xc6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c620 <unknown>

fdup z0.d, #18.00000000
// CHECK-INST: fmov z0.d, #18.00000000
// CHECK-ENCODING: [0x40,0xc6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c640 <unknown>

fdup z0.d, #19.00000000
// CHECK-INST: fmov z0.d, #19.00000000
// CHECK-ENCODING: [0x60,0xc6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c660 <unknown>

fdup z0.d, #20.00000000
// CHECK-INST: fmov z0.d, #20.00000000
// CHECK-ENCODING: [0x80,0xc6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c680 <unknown>

fdup z0.d, #21.00000000
// CHECK-INST: fmov z0.d, #21.00000000
// CHECK-ENCODING: [0xa0,0xc6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c6a0 <unknown>

fdup z0.d, #22.00000000
// CHECK-INST: fmov z0.d, #22.00000000
// CHECK-ENCODING: [0xc0,0xc6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c6c0 <unknown>

fdup z0.d, #23.00000000
// CHECK-INST: fmov z0.d, #23.00000000
// CHECK-ENCODING: [0xe0,0xc6,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c6e0 <unknown>

fdup z0.d, #24.00000000
// CHECK-INST: fmov z0.d, #24.00000000
// CHECK-ENCODING: [0x00,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c700 <unknown>

fdup z0.d, #25.00000000
// CHECK-INST: fmov z0.d, #25.00000000
// CHECK-ENCODING: [0x20,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c720 <unknown>

fdup z0.d, #26.00000000
// CHECK-INST: fmov z0.d, #26.00000000
// CHECK-ENCODING: [0x40,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c740 <unknown>

fdup z0.d, #27.00000000
// CHECK-INST: fmov z0.d, #27.00000000
// CHECK-ENCODING: [0x60,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c760 <unknown>

fdup z0.d, #28.00000000
// CHECK-INST: fmov z0.d, #28.00000000
// CHECK-ENCODING: [0x80,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c780 <unknown>

fdup z0.d, #29.00000000
// CHECK-INST: fmov z0.d, #29.00000000
// CHECK-ENCODING: [0xa0,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c7a0 <unknown>

fdup z0.d, #30.00000000
// CHECK-INST: fmov z0.d, #30.00000000
// CHECK-ENCODING: [0xc0,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c7c0 <unknown>

fdup z0.d, #31.00000000
// CHECK-INST: fmov z0.d, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xf9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25f9c7e0 <unknown>
