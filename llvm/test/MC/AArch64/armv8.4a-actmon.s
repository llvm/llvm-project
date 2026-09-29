// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s 2> %t | FileCheck %s --check-prefix=CHECK
// RUN: FileCheck --check-prefix=CHECK-RO < %t %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// ARMV8.4-A Activity Monitors
//------------------------------------------------------------------------------

msr AMCR_EL0, x0
msr AMCFGR_EL0, x0
msr AMCGCR_EL0, x0
msr AMUSERENR_EL0, x0
msr AMCNTENCLR0_EL0, x0
msr AMCNTENSET0_EL0, x0
msr AMEVCNTR00_EL0, x0
msr AMEVCNTR01_EL0, x0
msr AMEVCNTR02_EL0, x0
msr AMEVCNTR03_EL0, x0
msr AMEVTYPER00_EL0, x0
msr AMEVTYPER01_EL0, x0
msr AMEVTYPER02_EL0, x0
msr AMEVTYPER03_EL0, x0
msr AMCNTENCLR1_EL0, x0
msr AMCNTENSET1_EL0, x0
msr AMEVCNTR10_EL0, x0
msr AMEVCNTR11_EL0, x0
msr AMEVCNTR12_EL0, x0
msr AMEVCNTR13_EL0, x0
msr AMEVCNTR14_EL0, x0
msr AMEVCNTR15_EL0, x0
msr AMEVCNTR16_EL0, x0
msr AMEVCNTR17_EL0, x0
msr AMEVCNTR18_EL0, x0
msr AMEVCNTR19_EL0, x0
msr AMEVCNTR110_EL0, x0
msr AMEVCNTR111_EL0, x0
msr AMEVCNTR112_EL0, x0
msr AMEVCNTR113_EL0, x0
msr AMEVCNTR114_EL0, x0
msr AMEVCNTR115_EL0, x0
msr AMEVTYPER10_EL0, x0
msr AMEVTYPER11_EL0, x0
msr AMEVTYPER12_EL0, x0
msr AMEVTYPER13_EL0, x0
msr AMEVTYPER14_EL0, x0
msr AMEVTYPER15_EL0, x0
msr AMEVTYPER16_EL0, x0
msr AMEVTYPER17_EL0, x0
msr AMEVTYPER18_EL0, x0
msr AMEVTYPER19_EL0, x0
msr AMEVTYPER110_EL0, x0
msr AMEVTYPER111_EL0, x0
msr AMEVTYPER112_EL0, x0
msr AMEVTYPER113_EL0, x0
msr AMEVTYPER114_EL0, x0
msr AMEVTYPER115_EL0, x0

mrs x0, AMCR_EL0
mrs x0, AMCFGR_EL0
mrs x0, AMCGCR_EL0
mrs x0, AMUSERENR_EL0
mrs x0, AMCNTENCLR0_EL0
mrs x0, AMCNTENSET0_EL0
mrs x0, AMEVCNTR00_EL0
mrs x0, AMEVCNTR01_EL0
mrs x0, AMEVCNTR02_EL0
mrs x0, AMEVCNTR03_EL0
mrs x0, AMEVTYPER00_EL0
mrs x0, AMEVTYPER01_EL0
mrs x0, AMEVTYPER02_EL0
mrs x0, AMEVTYPER03_EL0
mrs x0, AMCNTENCLR1_EL0
mrs x0, AMCNTENSET1_EL0
mrs x0, AMEVCNTR10_EL0
mrs x0, AMEVCNTR11_EL0
mrs x0, AMEVCNTR12_EL0
mrs x0, AMEVCNTR13_EL0
mrs x0, AMEVCNTR14_EL0
mrs x0, AMEVCNTR15_EL0
mrs x0, AMEVCNTR16_EL0
mrs x0, AMEVCNTR17_EL0
mrs x0, AMEVCNTR18_EL0
mrs x0, AMEVCNTR19_EL0
mrs x0, AMEVCNTR110_EL0
mrs x0, AMEVCNTR111_EL0
mrs x0, AMEVCNTR112_EL0
mrs x0, AMEVCNTR113_EL0
mrs x0, AMEVCNTR114_EL0
mrs x0, AMEVCNTR115_EL0
mrs x0, AMEVTYPER10_EL0
mrs x0, AMEVTYPER11_EL0
mrs x0, AMEVTYPER12_EL0
mrs x0, AMEVTYPER13_EL0
mrs x0, AMEVTYPER14_EL0
mrs x0, AMEVTYPER15_EL0
mrs x0, AMEVTYPER16_EL0
mrs x0, AMEVTYPER17_EL0
mrs x0, AMEVTYPER18_EL0
mrs x0, AMEVTYPER19_EL0
mrs x0, AMEVTYPER110_EL0
mrs x0, AMEVTYPER111_EL0
mrs x0, AMEVTYPER112_EL0
mrs x0, AMEVTYPER113_EL0
mrs x0, AMEVTYPER114_EL0
mrs x0, AMEVTYPER115_EL0


//CHECK-RO: error: expected writable system register or pstate
//CHECK-RO: msr AMCFGR_EL0, x0
//CHECK-RO:     ^
//CHECK-RO: error: expected writable system register or pstate
//CHECK-RO: msr AMCGCR_EL0, x0
//CHECK-RO:     ^
//CHECK-RO: error: expected writable system register or pstate
//CHECK-RO: msr AMEVTYPER00_EL0, x0
//CHECK-RO:     ^
//CHECK-RO: error: expected writable system register or pstate
//CHECK-RO: msr AMEVTYPER01_EL0, x0
//CHECK-RO:     ^
//CHECK-RO: error: expected writable system register or pstate
//CHECK-RO: msr AMEVTYPER02_EL0, x0
//CHECK-RO:     ^
//CHECK-RO: error: expected writable system register or pstate
//CHECK-RO: msr AMEVTYPER03_EL0, x0
//CHECK-RO:     ^


//CHECK:  msr AMCR_EL0, x0            // encoding: [0x00,0xd2,0x1b,0xd5]
//CHECK:  msr AMUSERENR_EL0, x0       // encoding: [0x60,0xd2,0x1b,0xd5]
//CHECK:  msr AMCNTENCLR0_EL0, x0     // encoding: [0x80,0xd2,0x1b,0xd5]
//CHECK:  msr AMCNTENSET0_EL0, x0     // encoding: [0xa0,0xd2,0x1b,0xd5]
//CHECK:  msr AMEVCNTR00_EL0, x0      // encoding: [0x00,0xd4,0x1b,0xd5]
//CHECK:  msr AMEVCNTR01_EL0, x0      // encoding: [0x20,0xd4,0x1b,0xd5]
//CHECK:  msr AMEVCNTR02_EL0, x0      // encoding: [0x40,0xd4,0x1b,0xd5]
//CHECK:  msr AMEVCNTR03_EL0, x0      // encoding: [0x60,0xd4,0x1b,0xd5]
//CHECK:  msr AMCNTENCLR1_EL0, x0     // encoding: [0x00,0xd3,0x1b,0xd5]
//CHECK:  msr AMCNTENSET1_EL0, x0     // encoding: [0x20,0xd3,0x1b,0xd5]
//CHECK:  msr AMEVCNTR10_EL0, x0      // encoding: [0x00,0xdc,0x1b,0xd5]
//CHECK:  msr AMEVCNTR11_EL0, x0      // encoding: [0x20,0xdc,0x1b,0xd5]
//CHECK:  msr AMEVCNTR12_EL0, x0      // encoding: [0x40,0xdc,0x1b,0xd5]
//CHECK:  msr AMEVCNTR13_EL0, x0      // encoding: [0x60,0xdc,0x1b,0xd5]
//CHECK:  msr AMEVCNTR14_EL0, x0      // encoding: [0x80,0xdc,0x1b,0xd5]
//CHECK:  msr AMEVCNTR15_EL0, x0      // encoding: [0xa0,0xdc,0x1b,0xd5]
//CHECK:  msr AMEVCNTR16_EL0, x0      // encoding: [0xc0,0xdc,0x1b,0xd5]
//CHECK:  msr AMEVCNTR17_EL0, x0      // encoding: [0xe0,0xdc,0x1b,0xd5]
//CHECK:  msr AMEVCNTR18_EL0, x0      // encoding: [0x00,0xdd,0x1b,0xd5]
//CHECK:  msr AMEVCNTR19_EL0, x0      // encoding: [0x20,0xdd,0x1b,0xd5]
//CHECK:  msr AMEVCNTR110_EL0, x0     // encoding: [0x40,0xdd,0x1b,0xd5]
//CHECK:  msr AMEVCNTR111_EL0, x0     // encoding: [0x60,0xdd,0x1b,0xd5]
//CHECK:  msr AMEVCNTR112_EL0, x0     // encoding: [0x80,0xdd,0x1b,0xd5]
//CHECK:  msr AMEVCNTR113_EL0, x0     // encoding: [0xa0,0xdd,0x1b,0xd5]
//CHECK:  msr AMEVCNTR114_EL0, x0     // encoding: [0xc0,0xdd,0x1b,0xd5]
//CHECK:  msr AMEVCNTR115_EL0, x0     // encoding: [0xe0,0xdd,0x1b,0xd5]
//CHECK:  msr AMEVTYPER10_EL0, x0     // encoding: [0x00,0xde,0x1b,0xd5]
//CHECK:  msr AMEVTYPER11_EL0, x0     // encoding: [0x20,0xde,0x1b,0xd5]
//CHECK:  msr AMEVTYPER12_EL0, x0     // encoding: [0x40,0xde,0x1b,0xd5]
//CHECK:  msr AMEVTYPER13_EL0, x0     // encoding: [0x60,0xde,0x1b,0xd5]
//CHECK:  msr AMEVTYPER14_EL0, x0     // encoding: [0x80,0xde,0x1b,0xd5]
//CHECK:  msr AMEVTYPER15_EL0, x0     // encoding: [0xa0,0xde,0x1b,0xd5]
//CHECK:  msr AMEVTYPER16_EL0, x0     // encoding: [0xc0,0xde,0x1b,0xd5]
//CHECK:  msr AMEVTYPER17_EL0, x0     // encoding: [0xe0,0xde,0x1b,0xd5]
//CHECK:  msr AMEVTYPER18_EL0, x0     // encoding: [0x00,0xdf,0x1b,0xd5]
//CHECK:  msr AMEVTYPER19_EL0, x0     // encoding: [0x20,0xdf,0x1b,0xd5]
//CHECK:  msr AMEVTYPER110_EL0, x0    // encoding: [0x40,0xdf,0x1b,0xd5]
//CHECK:  msr AMEVTYPER111_EL0, x0    // encoding: [0x60,0xdf,0x1b,0xd5]
//CHECK:  msr AMEVTYPER112_EL0, x0    // encoding: [0x80,0xdf,0x1b,0xd5]
//CHECK:  msr AMEVTYPER113_EL0, x0    // encoding: [0xa0,0xdf,0x1b,0xd5]
//CHECK:  msr AMEVTYPER114_EL0, x0    // encoding: [0xc0,0xdf,0x1b,0xd5]
//CHECK:  msr AMEVTYPER115_EL0, x0    // encoding: [0xe0,0xdf,0x1b,0xd5]

//CHECK:  mrs x0, AMCR_EL0            // encoding: [0x00,0xd2,0x3b,0xd5]
//CHECK:  mrs x0, AMCFGR_EL0          // encoding: [0x20,0xd2,0x3b,0xd5]
//CHECK:  mrs x0, AMCGCR_EL0          // encoding: [0x40,0xd2,0x3b,0xd5]
//CHECK:  mrs x0, AMUSERENR_EL0       // encoding: [0x60,0xd2,0x3b,0xd5]
//CHECK:  mrs x0, AMCNTENCLR0_EL0     // encoding: [0x80,0xd2,0x3b,0xd5]
//CHECK:  mrs x0, AMCNTENSET0_EL0     // encoding: [0xa0,0xd2,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR00_EL0      // encoding: [0x00,0xd4,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR01_EL0      // encoding: [0x20,0xd4,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR02_EL0      // encoding: [0x40,0xd4,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR03_EL0      // encoding: [0x60,0xd4,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER00_EL0     // encoding: [0x00,0xd6,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER01_EL0     // encoding: [0x20,0xd6,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER02_EL0     // encoding: [0x40,0xd6,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER03_EL0     // encoding: [0x60,0xd6,0x3b,0xd5]
//CHECK:  mrs x0, AMCNTENCLR1_EL0     // encoding: [0x00,0xd3,0x3b,0xd5]
//CHECK:  mrs x0, AMCNTENSET1_EL0     // encoding: [0x20,0xd3,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR10_EL0      // encoding: [0x00,0xdc,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR11_EL0      // encoding: [0x20,0xdc,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR12_EL0      // encoding: [0x40,0xdc,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR13_EL0      // encoding: [0x60,0xdc,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR14_EL0      // encoding: [0x80,0xdc,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR15_EL0      // encoding: [0xa0,0xdc,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR16_EL0      // encoding: [0xc0,0xdc,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR17_EL0      // encoding: [0xe0,0xdc,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR18_EL0      // encoding: [0x00,0xdd,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR19_EL0      // encoding: [0x20,0xdd,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR110_EL0     // encoding: [0x40,0xdd,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR111_EL0     // encoding: [0x60,0xdd,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR112_EL0     // encoding: [0x80,0xdd,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR113_EL0     // encoding: [0xa0,0xdd,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR114_EL0     // encoding: [0xc0,0xdd,0x3b,0xd5]
//CHECK:  mrs x0, AMEVCNTR115_EL0     // encoding: [0xe0,0xdd,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER10_EL0     // encoding: [0x00,0xde,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER11_EL0     // encoding: [0x20,0xde,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER12_EL0     // encoding: [0x40,0xde,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER13_EL0     // encoding: [0x60,0xde,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER14_EL0     // encoding: [0x80,0xde,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER15_EL0     // encoding: [0xa0,0xde,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER16_EL0     // encoding: [0xc0,0xde,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER17_EL0     // encoding: [0xe0,0xde,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER18_EL0     // encoding: [0x00,0xdf,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER19_EL0     // encoding: [0x20,0xdf,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER110_EL0    // encoding: [0x40,0xdf,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER111_EL0    // encoding: [0x60,0xdf,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER112_EL0    // encoding: [0x80,0xdf,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER113_EL0    // encoding: [0xa0,0xdf,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER114_EL0    // encoding: [0xc0,0xdf,0x3b,0xd5]
//CHECK:  mrs x0, AMEVTYPER115_EL0    // encoding: [0xe0,0xdf,0x3b,0xd5]


//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMCR_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMCFGR_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMCGCR_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMUSERENR_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMCNTENCLR0_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMCNTENSET0_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR00_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR01_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR02_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR03_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER00_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER01_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER02_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER03_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMCNTENCLR1_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMCNTENSET1_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR10_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR11_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR12_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR13_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR14_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR15_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR16_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR17_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR18_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR19_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR110_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR111_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR112_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR113_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR114_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVCNTR115_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER10_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER11_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER12_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER13_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER14_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER15_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER16_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER17_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER18_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER19_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER110_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER111_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER112_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER113_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER114_EL0, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr AMEVTYPER115_EL0, x0
//CHECK-ERROR:     ^

//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMCR_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMCFGR_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMCGCR_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMUSERENR_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMCNTENCLR0_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMCNTENSET0_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR00_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR01_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR02_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR03_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER00_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER01_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER02_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER03_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMCNTENCLR1_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMCNTENSET1_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR10_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR11_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR12_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR13_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR14_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR15_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR16_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR17_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR18_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR19_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR110_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR111_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR112_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR113_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR114_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVCNTR115_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER10_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER11_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER12_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER13_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER14_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER15_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER16_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER17_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER18_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER19_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER110_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER111_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER112_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER113_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER114_EL0
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, AMEVTYPER115_EL0
//CHECK-ERROR:         ^
