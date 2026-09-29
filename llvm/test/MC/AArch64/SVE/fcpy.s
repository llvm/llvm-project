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

fcpy z0.h, p0/m, #-0.12500000
// CHECK-INST: fmov z0.h, p0/m, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0550d800 <unknown>

fcpy z0.s, p0/m, #-0.12500000
// CHECK-INST: fmov z0.s, p0/m, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0590d800 <unknown>

fcpy z0.d, p0/m, #-0.12500000
// CHECK-INST: fmov z0.d, p0/m, #-0.12500000
// CHECK-ENCODING: [0x00,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d800 <unknown>

fcpy z0.d, p0/m, #-0.13281250
// CHECK-INST: fmov z0.d, p0/m, #-0.13281250
// CHECK-ENCODING: [0x20,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d820 <unknown>

fcpy z0.d, p0/m, #-0.14062500
// CHECK-INST: fmov z0.d, p0/m, #-0.14062500
// CHECK-ENCODING: [0x40,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d840 <unknown>

fcpy z0.d, p0/m, #-0.14843750
// CHECK-INST: fmov z0.d, p0/m, #-0.14843750
// CHECK-ENCODING: [0x60,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d860 <unknown>

fcpy z0.d, p0/m, #-0.15625000
// CHECK-INST: fmov z0.d, p0/m, #-0.15625000
// CHECK-ENCODING: [0x80,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d880 <unknown>

fcpy z0.d, p0/m, #-0.16406250
// CHECK-INST: fmov z0.d, p0/m, #-0.16406250
// CHECK-ENCODING: [0xa0,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d8a0 <unknown>

fcpy z0.d, p0/m, #-0.17187500
// CHECK-INST: fmov z0.d, p0/m, #-0.17187500
// CHECK-ENCODING: [0xc0,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d8c0 <unknown>

fcpy z0.d, p0/m, #-0.17968750
// CHECK-INST: fmov z0.d, p0/m, #-0.17968750
// CHECK-ENCODING: [0xe0,0xd8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d8e0 <unknown>

fcpy z0.d, p0/m, #-0.18750000
// CHECK-INST: fmov z0.d, p0/m, #-0.18750000
// CHECK-ENCODING: [0x00,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d900 <unknown>

fcpy z0.d, p0/m, #-0.19531250
// CHECK-INST: fmov z0.d, p0/m, #-0.19531250
// CHECK-ENCODING: [0x20,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d920 <unknown>

fcpy z0.d, p0/m, #-0.20312500
// CHECK-INST: fmov z0.d, p0/m, #-0.20312500
// CHECK-ENCODING: [0x40,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d940 <unknown>

fcpy z0.d, p0/m, #-0.21093750
// CHECK-INST: fmov z0.d, p0/m, #-0.21093750
// CHECK-ENCODING: [0x60,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d960 <unknown>

fcpy z0.d, p0/m, #-0.21875000
// CHECK-INST: fmov z0.d, p0/m, #-0.21875000
// CHECK-ENCODING: [0x80,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d980 <unknown>

fcpy z0.d, p0/m, #-0.22656250
// CHECK-INST: fmov z0.d, p0/m, #-0.22656250
// CHECK-ENCODING: [0xa0,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d9a0 <unknown>

fcpy z0.d, p0/m, #-0.23437500
// CHECK-INST: fmov z0.d, p0/m, #-0.23437500
// CHECK-ENCODING: [0xc0,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d9c0 <unknown>

fcpy z0.d, p0/m, #-0.24218750
// CHECK-INST: fmov z0.d, p0/m, #-0.24218750
// CHECK-ENCODING: [0xe0,0xd9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d9e0 <unknown>

fcpy z0.d, p0/m, #-0.25000000
// CHECK-INST: fmov z0.d, p0/m, #-0.25000000
// CHECK-ENCODING: [0x00,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0da00 <unknown>

fcpy z0.d, p0/m, #-0.26562500
// CHECK-INST: fmov z0.d, p0/m, #-0.26562500
// CHECK-ENCODING: [0x20,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0da20 <unknown>

fcpy z0.d, p0/m, #-0.28125000
// CHECK-INST: fmov z0.d, p0/m, #-0.28125000
// CHECK-ENCODING: [0x40,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0da40 <unknown>

fcpy z0.d, p0/m, #-0.29687500
// CHECK-INST: fmov z0.d, p0/m, #-0.29687500
// CHECK-ENCODING: [0x60,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0da60 <unknown>

fcpy z0.d, p0/m, #-0.31250000
// CHECK-INST: fmov z0.d, p0/m, #-0.31250000
// CHECK-ENCODING: [0x80,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0da80 <unknown>

fcpy z0.d, p0/m, #-0.32812500
// CHECK-INST: fmov z0.d, p0/m, #-0.32812500
// CHECK-ENCODING: [0xa0,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0daa0 <unknown>

fcpy z0.d, p0/m, #-0.34375000
// CHECK-INST: fmov z0.d, p0/m, #-0.34375000
// CHECK-ENCODING: [0xc0,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dac0 <unknown>

fcpy z0.d, p0/m, #-0.35937500
// CHECK-INST: fmov z0.d, p0/m, #-0.35937500
// CHECK-ENCODING: [0xe0,0xda,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dae0 <unknown>

fcpy z0.d, p0/m, #-0.37500000
// CHECK-INST: fmov z0.d, p0/m, #-0.37500000
// CHECK-ENCODING: [0x00,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0db00 <unknown>

fcpy z0.d, p0/m, #-0.39062500
// CHECK-INST: fmov z0.d, p0/m, #-0.39062500
// CHECK-ENCODING: [0x20,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0db20 <unknown>

fcpy z0.d, p0/m, #-0.40625000
// CHECK-INST: fmov z0.d, p0/m, #-0.40625000
// CHECK-ENCODING: [0x40,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0db40 <unknown>

fcpy z0.d, p0/m, #-0.42187500
// CHECK-INST: fmov z0.d, p0/m, #-0.42187500
// CHECK-ENCODING: [0x60,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0db60 <unknown>

fcpy z0.d, p0/m, #-0.43750000
// CHECK-INST: fmov z0.d, p0/m, #-0.43750000
// CHECK-ENCODING: [0x80,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0db80 <unknown>

fcpy z0.d, p0/m, #-0.45312500
// CHECK-INST: fmov z0.d, p0/m, #-0.45312500
// CHECK-ENCODING: [0xa0,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dba0 <unknown>

fcpy z0.d, p0/m, #-0.46875000
// CHECK-INST: fmov z0.d, p0/m, #-0.46875000
// CHECK-ENCODING: [0xc0,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dbc0 <unknown>

fcpy z0.d, p0/m, #-0.48437500
// CHECK-INST: fmov z0.d, p0/m, #-0.48437500
// CHECK-ENCODING: [0xe0,0xdb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dbe0 <unknown>

fcpy z0.d, p0/m, #-0.50000000
// CHECK-INST: fmov z0.d, p0/m, #-0.50000000
// CHECK-ENCODING: [0x00,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dc00 <unknown>

fcpy z0.d, p0/m, #-0.53125000
// CHECK-INST: fmov z0.d, p0/m, #-0.53125000
// CHECK-ENCODING: [0x20,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dc20 <unknown>

fcpy z0.d, p0/m, #-0.56250000
// CHECK-INST: fmov z0.d, p0/m, #-0.56250000
// CHECK-ENCODING: [0x40,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dc40 <unknown>

fcpy z0.d, p0/m, #-0.59375000
// CHECK-INST: fmov z0.d, p0/m, #-0.59375000
// CHECK-ENCODING: [0x60,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dc60 <unknown>

fcpy z0.d, p0/m, #-0.62500000
// CHECK-INST: fmov z0.d, p0/m, #-0.62500000
// CHECK-ENCODING: [0x80,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dc80 <unknown>

fcpy z0.d, p0/m, #-0.65625000
// CHECK-INST: fmov z0.d, p0/m, #-0.65625000
// CHECK-ENCODING: [0xa0,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dca0 <unknown>

fcpy z0.d, p0/m, #-0.68750000
// CHECK-INST: fmov z0.d, p0/m, #-0.68750000
// CHECK-ENCODING: [0xc0,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dcc0 <unknown>

fcpy z0.d, p0/m, #-0.71875000
// CHECK-INST: fmov z0.d, p0/m, #-0.71875000
// CHECK-ENCODING: [0xe0,0xdc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dce0 <unknown>

fcpy z0.d, p0/m, #-0.75000000
// CHECK-INST: fmov z0.d, p0/m, #-0.75000000
// CHECK-ENCODING: [0x00,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dd00 <unknown>

fcpy z0.d, p0/m, #-0.78125000
// CHECK-INST: fmov z0.d, p0/m, #-0.78125000
// CHECK-ENCODING: [0x20,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dd20 <unknown>

fcpy z0.d, p0/m, #-0.81250000
// CHECK-INST: fmov z0.d, p0/m, #-0.81250000
// CHECK-ENCODING: [0x40,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dd40 <unknown>

fcpy z0.d, p0/m, #-0.84375000
// CHECK-INST: fmov z0.d, p0/m, #-0.84375000
// CHECK-ENCODING: [0x60,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dd60 <unknown>

fcpy z0.d, p0/m, #-0.87500000
// CHECK-INST: fmov z0.d, p0/m, #-0.87500000
// CHECK-ENCODING: [0x80,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dd80 <unknown>

fcpy z0.d, p0/m, #-0.90625000
// CHECK-INST: fmov z0.d, p0/m, #-0.90625000
// CHECK-ENCODING: [0xa0,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dda0 <unknown>

fcpy z0.d, p0/m, #-0.93750000
// CHECK-INST: fmov z0.d, p0/m, #-0.93750000
// CHECK-ENCODING: [0xc0,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ddc0 <unknown>

fcpy z0.d, p0/m, #-0.96875000
// CHECK-INST: fmov z0.d, p0/m, #-0.96875000
// CHECK-ENCODING: [0xe0,0xdd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dde0 <unknown>

fcpy z0.d, p0/m, #-1.00000000
// CHECK-INST: fmov z0.d, p0/m, #-1.00000000
// CHECK-ENCODING: [0x00,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0de00 <unknown>

fcpy z0.d, p0/m, #-1.06250000
// CHECK-INST: fmov z0.d, p0/m, #-1.06250000
// CHECK-ENCODING: [0x20,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0de20 <unknown>

fcpy z0.d, p0/m, #-1.12500000
// CHECK-INST: fmov z0.d, p0/m, #-1.12500000
// CHECK-ENCODING: [0x40,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0de40 <unknown>

fcpy z0.d, p0/m, #-1.18750000
// CHECK-INST: fmov z0.d, p0/m, #-1.18750000
// CHECK-ENCODING: [0x60,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0de60 <unknown>

fcpy z0.d, p0/m, #-1.25000000
// CHECK-INST: fmov z0.d, p0/m, #-1.25000000
// CHECK-ENCODING: [0x80,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0de80 <unknown>

fcpy z0.d, p0/m, #-1.31250000
// CHECK-INST: fmov z0.d, p0/m, #-1.31250000
// CHECK-ENCODING: [0xa0,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dea0 <unknown>

fcpy z0.d, p0/m, #-1.37500000
// CHECK-INST: fmov z0.d, p0/m, #-1.37500000
// CHECK-ENCODING: [0xc0,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dec0 <unknown>

fcpy z0.d, p0/m, #-1.43750000
// CHECK-INST: fmov z0.d, p0/m, #-1.43750000
// CHECK-ENCODING: [0xe0,0xde,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dee0 <unknown>

fcpy z0.d, p0/m, #-1.50000000
// CHECK-INST: fmov z0.d, p0/m, #-1.50000000
// CHECK-ENCODING: [0x00,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0df00 <unknown>

fcpy z0.d, p0/m, #-1.56250000
// CHECK-INST: fmov z0.d, p0/m, #-1.56250000
// CHECK-ENCODING: [0x20,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0df20 <unknown>

fcpy z0.d, p0/m, #-1.62500000
// CHECK-INST: fmov z0.d, p0/m, #-1.62500000
// CHECK-ENCODING: [0x40,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0df40 <unknown>

fcpy z0.d, p0/m, #-1.68750000
// CHECK-INST: fmov z0.d, p0/m, #-1.68750000
// CHECK-ENCODING: [0x60,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0df60 <unknown>

fcpy z0.d, p0/m, #-1.75000000
// CHECK-INST: fmov z0.d, p0/m, #-1.75000000
// CHECK-ENCODING: [0x80,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0df80 <unknown>

fcpy z0.d, p0/m, #-1.81250000
// CHECK-INST: fmov z0.d, p0/m, #-1.81250000
// CHECK-ENCODING: [0xa0,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dfa0 <unknown>

fcpy z0.d, p0/m, #-1.87500000
// CHECK-INST: fmov z0.d, p0/m, #-1.87500000
// CHECK-ENCODING: [0xc0,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dfc0 <unknown>

fcpy z0.d, p0/m, #-1.93750000
// CHECK-INST: fmov z0.d, p0/m, #-1.93750000
// CHECK-ENCODING: [0xe0,0xdf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0dfe0 <unknown>

fcpy z0.d, p0/m, #-2.00000000
// CHECK-INST: fmov z0.d, p0/m, #-2.00000000
// CHECK-ENCODING: [0x00,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d000 <unknown>

fcpy z0.d, p0/m, #-2.12500000
// CHECK-INST: fmov z0.d, p0/m, #-2.12500000
// CHECK-ENCODING: [0x20,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d020 <unknown>

fcpy z0.d, p0/m, #-2.25000000
// CHECK-INST: fmov z0.d, p0/m, #-2.25000000
// CHECK-ENCODING: [0x40,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d040 <unknown>

fcpy z0.d, p0/m, #-2.37500000
// CHECK-INST: fmov z0.d, p0/m, #-2.37500000
// CHECK-ENCODING: [0x60,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d060 <unknown>

fcpy z0.d, p0/m, #-2.50000000
// CHECK-INST: fmov z0.d, p0/m, #-2.50000000
// CHECK-ENCODING: [0x80,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d080 <unknown>

fcpy z0.d, p0/m, #-2.62500000
// CHECK-INST: fmov z0.d, p0/m, #-2.62500000
// CHECK-ENCODING: [0xa0,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d0a0 <unknown>

fcpy z0.d, p0/m, #-2.75000000
// CHECK-INST: fmov z0.d, p0/m, #-2.75000000
// CHECK-ENCODING: [0xc0,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d0c0 <unknown>

fcpy z0.d, p0/m, #-2.87500000
// CHECK-INST: fmov z0.d, p0/m, #-2.87500000
// CHECK-ENCODING: [0xe0,0xd0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d0e0 <unknown>

fcpy z0.d, p0/m, #-3.00000000
// CHECK-INST: fmov z0.d, p0/m, #-3.00000000
// CHECK-ENCODING: [0x00,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d100 <unknown>

fcpy z0.d, p0/m, #-3.12500000
// CHECK-INST: fmov z0.d, p0/m, #-3.12500000
// CHECK-ENCODING: [0x20,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d120 <unknown>

fcpy z0.d, p0/m, #-3.25000000
// CHECK-INST: fmov z0.d, p0/m, #-3.25000000
// CHECK-ENCODING: [0x40,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d140 <unknown>

fcpy z0.d, p0/m, #-3.37500000
// CHECK-INST: fmov z0.d, p0/m, #-3.37500000
// CHECK-ENCODING: [0x60,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d160 <unknown>

fcpy z0.d, p0/m, #-3.50000000
// CHECK-INST: fmov z0.d, p0/m, #-3.50000000
// CHECK-ENCODING: [0x80,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d180 <unknown>

fcpy z0.d, p0/m, #-3.62500000
// CHECK-INST: fmov z0.d, p0/m, #-3.62500000
// CHECK-ENCODING: [0xa0,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d1a0 <unknown>

fcpy z0.d, p0/m, #-3.75000000
// CHECK-INST: fmov z0.d, p0/m, #-3.75000000
// CHECK-ENCODING: [0xc0,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d1c0 <unknown>

fcpy z0.d, p0/m, #-3.87500000
// CHECK-INST: fmov z0.d, p0/m, #-3.87500000
// CHECK-ENCODING: [0xe0,0xd1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d1e0 <unknown>

fcpy z0.d, p0/m, #-4.00000000
// CHECK-INST: fmov z0.d, p0/m, #-4.00000000
// CHECK-ENCODING: [0x00,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d200 <unknown>

fcpy z0.d, p0/m, #-4.25000000
// CHECK-INST: fmov z0.d, p0/m, #-4.25000000
// CHECK-ENCODING: [0x20,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d220 <unknown>

fcpy z0.d, p0/m, #-4.50000000
// CHECK-INST: fmov z0.d, p0/m, #-4.50000000
// CHECK-ENCODING: [0x40,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d240 <unknown>

fcpy z0.d, p0/m, #-4.75000000
// CHECK-INST: fmov z0.d, p0/m, #-4.75000000
// CHECK-ENCODING: [0x60,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d260 <unknown>

fcpy z0.d, p0/m, #-5.00000000
// CHECK-INST: fmov z0.d, p0/m, #-5.00000000
// CHECK-ENCODING: [0x80,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d280 <unknown>

fcpy z0.d, p0/m, #-5.25000000
// CHECK-INST: fmov z0.d, p0/m, #-5.25000000
// CHECK-ENCODING: [0xa0,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d2a0 <unknown>

fcpy z0.d, p0/m, #-5.50000000
// CHECK-INST: fmov z0.d, p0/m, #-5.50000000
// CHECK-ENCODING: [0xc0,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d2c0 <unknown>

fcpy z0.d, p0/m, #-5.75000000
// CHECK-INST: fmov z0.d, p0/m, #-5.75000000
// CHECK-ENCODING: [0xe0,0xd2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d2e0 <unknown>

fcpy z0.d, p0/m, #-6.00000000
// CHECK-INST: fmov z0.d, p0/m, #-6.00000000
// CHECK-ENCODING: [0x00,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d300 <unknown>

fcpy z0.d, p0/m, #-6.25000000
// CHECK-INST: fmov z0.d, p0/m, #-6.25000000
// CHECK-ENCODING: [0x20,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d320 <unknown>

fcpy z0.d, p0/m, #-6.50000000
// CHECK-INST: fmov z0.d, p0/m, #-6.50000000
// CHECK-ENCODING: [0x40,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d340 <unknown>

fcpy z0.d, p0/m, #-6.75000000
// CHECK-INST: fmov z0.d, p0/m, #-6.75000000
// CHECK-ENCODING: [0x60,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d360 <unknown>

fcpy z0.d, p0/m, #-7.00000000
// CHECK-INST: fmov z0.d, p0/m, #-7.00000000
// CHECK-ENCODING: [0x80,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d380 <unknown>

fcpy z0.d, p0/m, #-7.25000000
// CHECK-INST: fmov z0.d, p0/m, #-7.25000000
// CHECK-ENCODING: [0xa0,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d3a0 <unknown>

fcpy z0.d, p0/m, #-7.50000000
// CHECK-INST: fmov z0.d, p0/m, #-7.50000000
// CHECK-ENCODING: [0xc0,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d3c0 <unknown>

fcpy z0.d, p0/m, #-7.75000000
// CHECK-INST: fmov z0.d, p0/m, #-7.75000000
// CHECK-ENCODING: [0xe0,0xd3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d3e0 <unknown>

fcpy z0.d, p0/m, #-8.00000000
// CHECK-INST: fmov z0.d, p0/m, #-8.00000000
// CHECK-ENCODING: [0x00,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d400 <unknown>

fcpy z0.d, p0/m, #-8.50000000
// CHECK-INST: fmov z0.d, p0/m, #-8.50000000
// CHECK-ENCODING: [0x20,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d420 <unknown>

fcpy z0.d, p0/m, #-9.00000000
// CHECK-INST: fmov z0.d, p0/m, #-9.00000000
// CHECK-ENCODING: [0x40,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d440 <unknown>

fcpy z0.d, p0/m, #-9.50000000
// CHECK-INST: fmov z0.d, p0/m, #-9.50000000
// CHECK-ENCODING: [0x60,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d460 <unknown>

fcpy z0.d, p0/m, #-10.00000000
// CHECK-INST: fmov z0.d, p0/m, #-10.00000000
// CHECK-ENCODING: [0x80,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d480 <unknown>

fcpy z0.d, p0/m, #-10.50000000
// CHECK-INST: fmov z0.d, p0/m, #-10.50000000
// CHECK-ENCODING: [0xa0,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d4a0 <unknown>

fcpy z0.d, p0/m, #-11.00000000
// CHECK-INST: fmov z0.d, p0/m, #-11.00000000
// CHECK-ENCODING: [0xc0,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d4c0 <unknown>

fcpy z0.d, p0/m, #-11.50000000
// CHECK-INST: fmov z0.d, p0/m, #-11.50000000
// CHECK-ENCODING: [0xe0,0xd4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d4e0 <unknown>

fcpy z0.d, p0/m, #-12.00000000
// CHECK-INST: fmov z0.d, p0/m, #-12.00000000
// CHECK-ENCODING: [0x00,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d500 <unknown>

fcpy z0.d, p0/m, #-12.50000000
// CHECK-INST: fmov z0.d, p0/m, #-12.50000000
// CHECK-ENCODING: [0x20,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d520 <unknown>

fcpy z0.d, p0/m, #-13.00000000
// CHECK-INST: fmov z0.d, p0/m, #-13.00000000
// CHECK-ENCODING: [0x40,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d540 <unknown>

fcpy z0.d, p0/m, #-13.50000000
// CHECK-INST: fmov z0.d, p0/m, #-13.50000000
// CHECK-ENCODING: [0x60,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d560 <unknown>

fcpy z0.d, p0/m, #-14.00000000
// CHECK-INST: fmov z0.d, p0/m, #-14.00000000
// CHECK-ENCODING: [0x80,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d580 <unknown>

fcpy z0.d, p0/m, #-14.50000000
// CHECK-INST: fmov z0.d, p0/m, #-14.50000000
// CHECK-ENCODING: [0xa0,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d5a0 <unknown>

fcpy z0.d, p0/m, #-15.00000000
// CHECK-INST: fmov z0.d, p0/m, #-15.00000000
// CHECK-ENCODING: [0xc0,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d5c0 <unknown>

fcpy z0.d, p0/m, #-15.50000000
// CHECK-INST: fmov z0.d, p0/m, #-15.50000000
// CHECK-ENCODING: [0xe0,0xd5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d5e0 <unknown>

fcpy z0.d, p0/m, #-16.00000000
// CHECK-INST: fmov z0.d, p0/m, #-16.00000000
// CHECK-ENCODING: [0x00,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d600 <unknown>

fcpy z0.d, p0/m, #-17.00000000
// CHECK-INST: fmov z0.d, p0/m, #-17.00000000
// CHECK-ENCODING: [0x20,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d620 <unknown>

fcpy z0.d, p0/m, #-18.00000000
// CHECK-INST: fmov z0.d, p0/m, #-18.00000000
// CHECK-ENCODING: [0x40,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d640 <unknown>

fcpy z0.d, p0/m, #-19.00000000
// CHECK-INST: fmov z0.d, p0/m, #-19.00000000
// CHECK-ENCODING: [0x60,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d660 <unknown>

fcpy z0.d, p0/m, #-20.00000000
// CHECK-INST: fmov z0.d, p0/m, #-20.00000000
// CHECK-ENCODING: [0x80,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d680 <unknown>

fcpy z0.d, p0/m, #-21.00000000
// CHECK-INST: fmov z0.d, p0/m, #-21.00000000
// CHECK-ENCODING: [0xa0,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d6a0 <unknown>

fcpy z0.d, p0/m, #-22.00000000
// CHECK-INST: fmov z0.d, p0/m, #-22.00000000
// CHECK-ENCODING: [0xc0,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d6c0 <unknown>

fcpy z0.d, p0/m, #-23.00000000
// CHECK-INST: fmov z0.d, p0/m, #-23.00000000
// CHECK-ENCODING: [0xe0,0xd6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d6e0 <unknown>

fcpy z0.d, p0/m, #-24.00000000
// CHECK-INST: fmov z0.d, p0/m, #-24.00000000
// CHECK-ENCODING: [0x00,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d700 <unknown>

fcpy z0.d, p0/m, #-25.00000000
// CHECK-INST: fmov z0.d, p0/m, #-25.00000000
// CHECK-ENCODING: [0x20,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d720 <unknown>

fcpy z0.d, p0/m, #-26.00000000
// CHECK-INST: fmov z0.d, p0/m, #-26.00000000
// CHECK-ENCODING: [0x40,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d740 <unknown>

fcpy z0.d, p0/m, #-27.00000000
// CHECK-INST: fmov z0.d, p0/m, #-27.00000000
// CHECK-ENCODING: [0x60,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d760 <unknown>

fcpy z0.d, p0/m, #-28.00000000
// CHECK-INST: fmov z0.d, p0/m, #-28.00000000
// CHECK-ENCODING: [0x80,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d780 <unknown>

fcpy z0.d, p0/m, #-29.00000000
// CHECK-INST: fmov z0.d, p0/m, #-29.00000000
// CHECK-ENCODING: [0xa0,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d7a0 <unknown>

fcpy z0.d, p0/m, #-30.00000000
// CHECK-INST: fmov z0.d, p0/m, #-30.00000000
// CHECK-ENCODING: [0xc0,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d7c0 <unknown>

fcpy z0.d, p0/m, #-31.00000000
// CHECK-INST: fmov z0.d, p0/m, #-31.00000000
// CHECK-ENCODING: [0xe0,0xd7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0d7e0 <unknown>

fcpy z0.d, p0/m, #0.12500000
// CHECK-INST: fmov z0.d, p0/m, #0.12500000
// CHECK-ENCODING: [0x00,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c800 <unknown>

fcpy z0.d, p0/m, #0.13281250
// CHECK-INST: fmov z0.d, p0/m, #0.13281250
// CHECK-ENCODING: [0x20,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c820 <unknown>

fcpy z0.d, p0/m, #0.14062500
// CHECK-INST: fmov z0.d, p0/m, #0.14062500
// CHECK-ENCODING: [0x40,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c840 <unknown>

fcpy z0.d, p0/m, #0.14843750
// CHECK-INST: fmov z0.d, p0/m, #0.14843750
// CHECK-ENCODING: [0x60,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c860 <unknown>

fcpy z0.d, p0/m, #0.15625000
// CHECK-INST: fmov z0.d, p0/m, #0.15625000
// CHECK-ENCODING: [0x80,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c880 <unknown>

fcpy z0.d, p0/m, #0.16406250
// CHECK-INST: fmov z0.d, p0/m, #0.16406250
// CHECK-ENCODING: [0xa0,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c8a0 <unknown>

fcpy z0.d, p0/m, #0.17187500
// CHECK-INST: fmov z0.d, p0/m, #0.17187500
// CHECK-ENCODING: [0xc0,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c8c0 <unknown>

fcpy z0.d, p0/m, #0.17968750
// CHECK-INST: fmov z0.d, p0/m, #0.17968750
// CHECK-ENCODING: [0xe0,0xc8,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c8e0 <unknown>

fcpy z0.d, p0/m, #0.18750000
// CHECK-INST: fmov z0.d, p0/m, #0.18750000
// CHECK-ENCODING: [0x00,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c900 <unknown>

fcpy z0.d, p0/m, #0.19531250
// CHECK-INST: fmov z0.d, p0/m, #0.19531250
// CHECK-ENCODING: [0x20,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c920 <unknown>

fcpy z0.d, p0/m, #0.20312500
// CHECK-INST: fmov z0.d, p0/m, #0.20312500
// CHECK-ENCODING: [0x40,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c940 <unknown>

fcpy z0.d, p0/m, #0.21093750
// CHECK-INST: fmov z0.d, p0/m, #0.21093750
// CHECK-ENCODING: [0x60,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c960 <unknown>

fcpy z0.d, p0/m, #0.21875000
// CHECK-INST: fmov z0.d, p0/m, #0.21875000
// CHECK-ENCODING: [0x80,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c980 <unknown>

fcpy z0.d, p0/m, #0.22656250
// CHECK-INST: fmov z0.d, p0/m, #0.22656250
// CHECK-ENCODING: [0xa0,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c9a0 <unknown>

fcpy z0.d, p0/m, #0.23437500
// CHECK-INST: fmov z0.d, p0/m, #0.23437500
// CHECK-ENCODING: [0xc0,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c9c0 <unknown>

fcpy z0.d, p0/m, #0.24218750
// CHECK-INST: fmov z0.d, p0/m, #0.24218750
// CHECK-ENCODING: [0xe0,0xc9,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c9e0 <unknown>

fcpy z0.d, p0/m, #0.25000000
// CHECK-INST: fmov z0.d, p0/m, #0.25000000
// CHECK-ENCODING: [0x00,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ca00 <unknown>

fcpy z0.d, p0/m, #0.26562500
// CHECK-INST: fmov z0.d, p0/m, #0.26562500
// CHECK-ENCODING: [0x20,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ca20 <unknown>

fcpy z0.d, p0/m, #0.28125000
// CHECK-INST: fmov z0.d, p0/m, #0.28125000
// CHECK-ENCODING: [0x40,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ca40 <unknown>

fcpy z0.d, p0/m, #0.29687500
// CHECK-INST: fmov z0.d, p0/m, #0.29687500
// CHECK-ENCODING: [0x60,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ca60 <unknown>

fcpy z0.d, p0/m, #0.31250000
// CHECK-INST: fmov z0.d, p0/m, #0.31250000
// CHECK-ENCODING: [0x80,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ca80 <unknown>

fcpy z0.d, p0/m, #0.32812500
// CHECK-INST: fmov z0.d, p0/m, #0.32812500
// CHECK-ENCODING: [0xa0,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0caa0 <unknown>

fcpy z0.d, p0/m, #0.34375000
// CHECK-INST: fmov z0.d, p0/m, #0.34375000
// CHECK-ENCODING: [0xc0,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cac0 <unknown>

fcpy z0.d, p0/m, #0.35937500
// CHECK-INST: fmov z0.d, p0/m, #0.35937500
// CHECK-ENCODING: [0xe0,0xca,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cae0 <unknown>

fcpy z0.d, p0/m, #0.37500000
// CHECK-INST: fmov z0.d, p0/m, #0.37500000
// CHECK-ENCODING: [0x00,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cb00 <unknown>

fcpy z0.d, p0/m, #0.39062500
// CHECK-INST: fmov z0.d, p0/m, #0.39062500
// CHECK-ENCODING: [0x20,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cb20 <unknown>

fcpy z0.d, p0/m, #0.40625000
// CHECK-INST: fmov z0.d, p0/m, #0.40625000
// CHECK-ENCODING: [0x40,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cb40 <unknown>

fcpy z0.d, p0/m, #0.42187500
// CHECK-INST: fmov z0.d, p0/m, #0.42187500
// CHECK-ENCODING: [0x60,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cb60 <unknown>

fcpy z0.d, p0/m, #0.43750000
// CHECK-INST: fmov z0.d, p0/m, #0.43750000
// CHECK-ENCODING: [0x80,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cb80 <unknown>

fcpy z0.d, p0/m, #0.45312500
// CHECK-INST: fmov z0.d, p0/m, #0.45312500
// CHECK-ENCODING: [0xa0,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cba0 <unknown>

fcpy z0.d, p0/m, #0.46875000
// CHECK-INST: fmov z0.d, p0/m, #0.46875000
// CHECK-ENCODING: [0xc0,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cbc0 <unknown>

fcpy z0.d, p0/m, #0.48437500
// CHECK-INST: fmov z0.d, p0/m, #0.48437500
// CHECK-ENCODING: [0xe0,0xcb,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cbe0 <unknown>

fcpy z0.d, p0/m, #0.50000000
// CHECK-INST: fmov z0.d, p0/m, #0.50000000
// CHECK-ENCODING: [0x00,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cc00 <unknown>

fcpy z0.d, p0/m, #0.53125000
// CHECK-INST: fmov z0.d, p0/m, #0.53125000
// CHECK-ENCODING: [0x20,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cc20 <unknown>

fcpy z0.d, p0/m, #0.56250000
// CHECK-INST: fmov z0.d, p0/m, #0.56250000
// CHECK-ENCODING: [0x40,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cc40 <unknown>

fcpy z0.d, p0/m, #0.59375000
// CHECK-INST: fmov z0.d, p0/m, #0.59375000
// CHECK-ENCODING: [0x60,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cc60 <unknown>

fcpy z0.d, p0/m, #0.62500000
// CHECK-INST: fmov z0.d, p0/m, #0.62500000
// CHECK-ENCODING: [0x80,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cc80 <unknown>

fcpy z0.d, p0/m, #0.65625000
// CHECK-INST: fmov z0.d, p0/m, #0.65625000
// CHECK-ENCODING: [0xa0,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cca0 <unknown>

fcpy z0.d, p0/m, #0.68750000
// CHECK-INST: fmov z0.d, p0/m, #0.68750000
// CHECK-ENCODING: [0xc0,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ccc0 <unknown>

fcpy z0.d, p0/m, #0.71875000
// CHECK-INST: fmov z0.d, p0/m, #0.71875000
// CHECK-ENCODING: [0xe0,0xcc,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cce0 <unknown>

fcpy z0.d, p0/m, #0.75000000
// CHECK-INST: fmov z0.d, p0/m, #0.75000000
// CHECK-ENCODING: [0x00,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cd00 <unknown>

fcpy z0.d, p0/m, #0.78125000
// CHECK-INST: fmov z0.d, p0/m, #0.78125000
// CHECK-ENCODING: [0x20,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cd20 <unknown>

fcpy z0.d, p0/m, #0.81250000
// CHECK-INST: fmov z0.d, p0/m, #0.81250000
// CHECK-ENCODING: [0x40,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cd40 <unknown>

fcpy z0.d, p0/m, #0.84375000
// CHECK-INST: fmov z0.d, p0/m, #0.84375000
// CHECK-ENCODING: [0x60,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cd60 <unknown>

fcpy z0.d, p0/m, #0.87500000
// CHECK-INST: fmov z0.d, p0/m, #0.87500000
// CHECK-ENCODING: [0x80,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cd80 <unknown>

fcpy z0.d, p0/m, #0.90625000
// CHECK-INST: fmov z0.d, p0/m, #0.90625000
// CHECK-ENCODING: [0xa0,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cda0 <unknown>

fcpy z0.d, p0/m, #0.93750000
// CHECK-INST: fmov z0.d, p0/m, #0.93750000
// CHECK-ENCODING: [0xc0,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cdc0 <unknown>

fcpy z0.d, p0/m, #0.96875000
// CHECK-INST: fmov z0.d, p0/m, #0.96875000
// CHECK-ENCODING: [0xe0,0xcd,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cde0 <unknown>

fcpy z0.d, p0/m, #1.00000000
// CHECK-INST: fmov z0.d, p0/m, #1.00000000
// CHECK-ENCODING: [0x00,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ce00 <unknown>

fcpy z0.d, p0/m, #1.06250000
// CHECK-INST: fmov z0.d, p0/m, #1.06250000
// CHECK-ENCODING: [0x20,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ce20 <unknown>

fcpy z0.d, p0/m, #1.12500000
// CHECK-INST: fmov z0.d, p0/m, #1.12500000
// CHECK-ENCODING: [0x40,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ce40 <unknown>

fcpy z0.d, p0/m, #1.18750000
// CHECK-INST: fmov z0.d, p0/m, #1.18750000
// CHECK-ENCODING: [0x60,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ce60 <unknown>

fcpy z0.d, p0/m, #1.25000000
// CHECK-INST: fmov z0.d, p0/m, #1.25000000
// CHECK-ENCODING: [0x80,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0ce80 <unknown>

fcpy z0.d, p0/m, #1.31250000
// CHECK-INST: fmov z0.d, p0/m, #1.31250000
// CHECK-ENCODING: [0xa0,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cea0 <unknown>

fcpy z0.d, p0/m, #1.37500000
// CHECK-INST: fmov z0.d, p0/m, #1.37500000
// CHECK-ENCODING: [0xc0,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cec0 <unknown>

fcpy z0.d, p0/m, #1.43750000
// CHECK-INST: fmov z0.d, p0/m, #1.43750000
// CHECK-ENCODING: [0xe0,0xce,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cee0 <unknown>

fcpy z0.d, p0/m, #1.50000000
// CHECK-INST: fmov z0.d, p0/m, #1.50000000
// CHECK-ENCODING: [0x00,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cf00 <unknown>

fcpy z0.d, p0/m, #1.56250000
// CHECK-INST: fmov z0.d, p0/m, #1.56250000
// CHECK-ENCODING: [0x20,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cf20 <unknown>

fcpy z0.d, p0/m, #1.62500000
// CHECK-INST: fmov z0.d, p0/m, #1.62500000
// CHECK-ENCODING: [0x40,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cf40 <unknown>

fcpy z0.d, p0/m, #1.68750000
// CHECK-INST: fmov z0.d, p0/m, #1.68750000
// CHECK-ENCODING: [0x60,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cf60 <unknown>

fcpy z0.d, p0/m, #1.75000000
// CHECK-INST: fmov z0.d, p0/m, #1.75000000
// CHECK-ENCODING: [0x80,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cf80 <unknown>

fcpy z0.d, p0/m, #1.81250000
// CHECK-INST: fmov z0.d, p0/m, #1.81250000
// CHECK-ENCODING: [0xa0,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cfa0 <unknown>

fcpy z0.d, p0/m, #1.87500000
// CHECK-INST: fmov z0.d, p0/m, #1.87500000
// CHECK-ENCODING: [0xc0,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cfc0 <unknown>

fcpy z0.d, p0/m, #1.93750000
// CHECK-INST: fmov z0.d, p0/m, #1.93750000
// CHECK-ENCODING: [0xe0,0xcf,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0cfe0 <unknown>

fcpy z0.d, p0/m, #2.00000000
// CHECK-INST: fmov z0.d, p0/m, #2.00000000
// CHECK-ENCODING: [0x00,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c000 <unknown>

fcpy z0.d, p0/m, #2.12500000
// CHECK-INST: fmov z0.d, p0/m, #2.12500000
// CHECK-ENCODING: [0x20,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c020 <unknown>

fcpy z0.d, p0/m, #2.25000000
// CHECK-INST: fmov z0.d, p0/m, #2.25000000
// CHECK-ENCODING: [0x40,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c040 <unknown>

fcpy z0.d, p0/m, #2.37500000
// CHECK-INST: fmov z0.d, p0/m, #2.37500000
// CHECK-ENCODING: [0x60,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c060 <unknown>

fcpy z0.d, p0/m, #2.50000000
// CHECK-INST: fmov z0.d, p0/m, #2.50000000
// CHECK-ENCODING: [0x80,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c080 <unknown>

fcpy z0.d, p0/m, #2.62500000
// CHECK-INST: fmov z0.d, p0/m, #2.62500000
// CHECK-ENCODING: [0xa0,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c0a0 <unknown>

fcpy z0.d, p0/m, #2.75000000
// CHECK-INST: fmov z0.d, p0/m, #2.75000000
// CHECK-ENCODING: [0xc0,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c0c0 <unknown>

fcpy z0.d, p0/m, #2.87500000
// CHECK-INST: fmov z0.d, p0/m, #2.87500000
// CHECK-ENCODING: [0xe0,0xc0,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c0e0 <unknown>

fcpy z0.d, p0/m, #3.00000000
// CHECK-INST: fmov z0.d, p0/m, #3.00000000
// CHECK-ENCODING: [0x00,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c100 <unknown>

fcpy z0.d, p0/m, #3.12500000
// CHECK-INST: fmov z0.d, p0/m, #3.12500000
// CHECK-ENCODING: [0x20,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c120 <unknown>

fcpy z0.d, p0/m, #3.25000000
// CHECK-INST: fmov z0.d, p0/m, #3.25000000
// CHECK-ENCODING: [0x40,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c140 <unknown>

fcpy z0.d, p0/m, #3.37500000
// CHECK-INST: fmov z0.d, p0/m, #3.37500000
// CHECK-ENCODING: [0x60,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c160 <unknown>

fcpy z0.d, p0/m, #3.50000000
// CHECK-INST: fmov z0.d, p0/m, #3.50000000
// CHECK-ENCODING: [0x80,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c180 <unknown>

fcpy z0.d, p0/m, #3.62500000
// CHECK-INST: fmov z0.d, p0/m, #3.62500000
// CHECK-ENCODING: [0xa0,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c1a0 <unknown>

fcpy z0.d, p0/m, #3.75000000
// CHECK-INST: fmov z0.d, p0/m, #3.75000000
// CHECK-ENCODING: [0xc0,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c1c0 <unknown>

fcpy z0.d, p0/m, #3.87500000
// CHECK-INST: fmov z0.d, p0/m, #3.87500000
// CHECK-ENCODING: [0xe0,0xc1,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c1e0 <unknown>

fcpy z0.d, p0/m, #4.00000000
// CHECK-INST: fmov z0.d, p0/m, #4.00000000
// CHECK-ENCODING: [0x00,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c200 <unknown>

fcpy z0.d, p0/m, #4.25000000
// CHECK-INST: fmov z0.d, p0/m, #4.25000000
// CHECK-ENCODING: [0x20,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c220 <unknown>

fcpy z0.d, p0/m, #4.50000000
// CHECK-INST: fmov z0.d, p0/m, #4.50000000
// CHECK-ENCODING: [0x40,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c240 <unknown>

fcpy z0.d, p0/m, #4.75000000
// CHECK-INST: fmov z0.d, p0/m, #4.75000000
// CHECK-ENCODING: [0x60,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c260 <unknown>

fcpy z0.d, p0/m, #5.00000000
// CHECK-INST: fmov z0.d, p0/m, #5.00000000
// CHECK-ENCODING: [0x80,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c280 <unknown>

fcpy z0.d, p0/m, #5.25000000
// CHECK-INST: fmov z0.d, p0/m, #5.25000000
// CHECK-ENCODING: [0xa0,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c2a0 <unknown>

fcpy z0.d, p0/m, #5.50000000
// CHECK-INST: fmov z0.d, p0/m, #5.50000000
// CHECK-ENCODING: [0xc0,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c2c0 <unknown>

fcpy z0.d, p0/m, #5.75000000
// CHECK-INST: fmov z0.d, p0/m, #5.75000000
// CHECK-ENCODING: [0xe0,0xc2,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c2e0 <unknown>

fcpy z0.d, p0/m, #6.00000000
// CHECK-INST: fmov z0.d, p0/m, #6.00000000
// CHECK-ENCODING: [0x00,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c300 <unknown>

fcpy z0.d, p0/m, #6.25000000
// CHECK-INST: fmov z0.d, p0/m, #6.25000000
// CHECK-ENCODING: [0x20,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c320 <unknown>

fcpy z0.d, p0/m, #6.50000000
// CHECK-INST: fmov z0.d, p0/m, #6.50000000
// CHECK-ENCODING: [0x40,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c340 <unknown>

fcpy z0.d, p0/m, #6.75000000
// CHECK-INST: fmov z0.d, p0/m, #6.75000000
// CHECK-ENCODING: [0x60,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c360 <unknown>

fcpy z0.d, p0/m, #7.00000000
// CHECK-INST: fmov z0.d, p0/m, #7.00000000
// CHECK-ENCODING: [0x80,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c380 <unknown>

fcpy z0.d, p0/m, #7.25000000
// CHECK-INST: fmov z0.d, p0/m, #7.25000000
// CHECK-ENCODING: [0xa0,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c3a0 <unknown>

fcpy z0.d, p0/m, #7.50000000
// CHECK-INST: fmov z0.d, p0/m, #7.50000000
// CHECK-ENCODING: [0xc0,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c3c0 <unknown>

fcpy z0.d, p0/m, #7.75000000
// CHECK-INST: fmov z0.d, p0/m, #7.75000000
// CHECK-ENCODING: [0xe0,0xc3,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c3e0 <unknown>

fcpy z0.d, p0/m, #8.00000000
// CHECK-INST: fmov z0.d, p0/m, #8.00000000
// CHECK-ENCODING: [0x00,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c400 <unknown>

fcpy z0.d, p0/m, #8.50000000
// CHECK-INST: fmov z0.d, p0/m, #8.50000000
// CHECK-ENCODING: [0x20,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c420 <unknown>

fcpy z0.d, p0/m, #9.00000000
// CHECK-INST: fmov z0.d, p0/m, #9.00000000
// CHECK-ENCODING: [0x40,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c440 <unknown>

fcpy z0.d, p0/m, #9.50000000
// CHECK-INST: fmov z0.d, p0/m, #9.50000000
// CHECK-ENCODING: [0x60,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c460 <unknown>

fcpy z0.d, p0/m, #10.00000000
// CHECK-INST: fmov z0.d, p0/m, #10.00000000
// CHECK-ENCODING: [0x80,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c480 <unknown>

fcpy z0.d, p0/m, #10.50000000
// CHECK-INST: fmov z0.d, p0/m, #10.50000000
// CHECK-ENCODING: [0xa0,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c4a0 <unknown>

fcpy z0.d, p0/m, #11.00000000
// CHECK-INST: fmov z0.d, p0/m, #11.00000000
// CHECK-ENCODING: [0xc0,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c4c0 <unknown>

fcpy z0.d, p0/m, #11.50000000
// CHECK-INST: fmov z0.d, p0/m, #11.50000000
// CHECK-ENCODING: [0xe0,0xc4,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c4e0 <unknown>

fcpy z0.d, p0/m, #12.00000000
// CHECK-INST: fmov z0.d, p0/m, #12.00000000
// CHECK-ENCODING: [0x00,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c500 <unknown>

fcpy z0.d, p0/m, #12.50000000
// CHECK-INST: fmov z0.d, p0/m, #12.50000000
// CHECK-ENCODING: [0x20,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c520 <unknown>

fcpy z0.d, p0/m, #13.00000000
// CHECK-INST: fmov z0.d, p0/m, #13.00000000
// CHECK-ENCODING: [0x40,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c540 <unknown>

fcpy z0.d, p0/m, #13.50000000
// CHECK-INST: fmov z0.d, p0/m, #13.50000000
// CHECK-ENCODING: [0x60,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c560 <unknown>

fcpy z0.d, p0/m, #14.00000000
// CHECK-INST: fmov z0.d, p0/m, #14.00000000
// CHECK-ENCODING: [0x80,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c580 <unknown>

fcpy z0.d, p0/m, #14.50000000
// CHECK-INST: fmov z0.d, p0/m, #14.50000000
// CHECK-ENCODING: [0xa0,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c5a0 <unknown>

fcpy z0.d, p0/m, #15.00000000
// CHECK-INST: fmov z0.d, p0/m, #15.00000000
// CHECK-ENCODING: [0xc0,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c5c0 <unknown>

fcpy z0.d, p0/m, #15.50000000
// CHECK-INST: fmov z0.d, p0/m, #15.50000000
// CHECK-ENCODING: [0xe0,0xc5,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c5e0 <unknown>

fcpy z0.d, p0/m, #16.00000000
// CHECK-INST: fmov z0.d, p0/m, #16.00000000
// CHECK-ENCODING: [0x00,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c600 <unknown>

fcpy z0.d, p0/m, #17.00000000
// CHECK-INST: fmov z0.d, p0/m, #17.00000000
// CHECK-ENCODING: [0x20,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c620 <unknown>

fcpy z0.d, p0/m, #18.00000000
// CHECK-INST: fmov z0.d, p0/m, #18.00000000
// CHECK-ENCODING: [0x40,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c640 <unknown>

fcpy z0.d, p0/m, #19.00000000
// CHECK-INST: fmov z0.d, p0/m, #19.00000000
// CHECK-ENCODING: [0x60,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c660 <unknown>

fcpy z0.d, p0/m, #20.00000000
// CHECK-INST: fmov z0.d, p0/m, #20.00000000
// CHECK-ENCODING: [0x80,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c680 <unknown>

fcpy z0.d, p0/m, #21.00000000
// CHECK-INST: fmov z0.d, p0/m, #21.00000000
// CHECK-ENCODING: [0xa0,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c6a0 <unknown>

fcpy z0.d, p0/m, #22.00000000
// CHECK-INST: fmov z0.d, p0/m, #22.00000000
// CHECK-ENCODING: [0xc0,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c6c0 <unknown>

fcpy z0.d, p0/m, #23.00000000
// CHECK-INST: fmov z0.d, p0/m, #23.00000000
// CHECK-ENCODING: [0xe0,0xc6,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c6e0 <unknown>

fcpy z0.d, p0/m, #24.00000000
// CHECK-INST: fmov z0.d, p0/m, #24.00000000
// CHECK-ENCODING: [0x00,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c700 <unknown>

fcpy z0.d, p0/m, #25.00000000
// CHECK-INST: fmov z0.d, p0/m, #25.00000000
// CHECK-ENCODING: [0x20,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c720 <unknown>

fcpy z0.d, p0/m, #26.00000000
// CHECK-INST: fmov z0.d, p0/m, #26.00000000
// CHECK-ENCODING: [0x40,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c740 <unknown>

fcpy z0.d, p0/m, #27.00000000
// CHECK-INST: fmov z0.d, p0/m, #27.00000000
// CHECK-ENCODING: [0x60,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c760 <unknown>

fcpy z0.d, p0/m, #28.00000000
// CHECK-INST: fmov z0.d, p0/m, #28.00000000
// CHECK-ENCODING: [0x80,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c780 <unknown>

fcpy z0.d, p0/m, #29.00000000
// CHECK-INST: fmov z0.d, p0/m, #29.00000000
// CHECK-ENCODING: [0xa0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c7a0 <unknown>

fcpy z0.d, p0/m, #30.00000000
// CHECK-INST: fmov z0.d, p0/m, #30.00000000
// CHECK-ENCODING: [0xc0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c7c0 <unknown>

fcpy z0.d, p0/m, #31.00000000
// CHECK-INST: fmov z0.d, p0/m, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c7e0 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0.d, p0/z, z7.d
// CHECK-INST: movprfx	z0.d, p0/z, z7.d
// CHECK-ENCODING: [0xe0,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020e0 <unknown>

fcpy z0.d, p0/m, #31.00000000
// CHECK-INST: fmov	z0.d, p0/m, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c7e0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

fcpy z0.d, p0/m, #31.00000000
// CHECK-INST: fmov	z0.d, p0/m, #31.00000000
// CHECK-ENCODING: [0xe0,0xc7,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d0c7e0 <unknown>
