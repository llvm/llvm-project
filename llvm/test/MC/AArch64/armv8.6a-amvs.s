// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+amvs -o - %s  | FileCheck %s
// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+v8.6a -o - %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -o - %s 2>&1  | FileCheck %s --check-prefix=CHECK-ERROR
mrs x0, AMCG1IDR_EL0
msr AMEVCNTVOFF00_EL2, x0
msr AMEVCNTVOFF01_EL2, x0
msr AMEVCNTVOFF02_EL2, x0
msr AMEVCNTVOFF03_EL2, x0
msr AMEVCNTVOFF04_EL2, x0
msr AMEVCNTVOFF05_EL2, x0
msr AMEVCNTVOFF06_EL2, x0
msr AMEVCNTVOFF07_EL2, x0
msr AMEVCNTVOFF08_EL2, x0
msr AMEVCNTVOFF09_EL2, x0
msr AMEVCNTVOFF010_EL2, x0
msr AMEVCNTVOFF011_EL2, x0
msr AMEVCNTVOFF012_EL2, x0
msr AMEVCNTVOFF013_EL2, x0
msr AMEVCNTVOFF014_EL2, x0
msr AMEVCNTVOFF015_EL2, x0
mrs x0, AMEVCNTVOFF00_EL2
mrs x0, AMEVCNTVOFF01_EL2
mrs x0, AMEVCNTVOFF02_EL2
mrs x0, AMEVCNTVOFF03_EL2
mrs x0, AMEVCNTVOFF04_EL2
mrs x0, AMEVCNTVOFF05_EL2
mrs x0, AMEVCNTVOFF06_EL2
mrs x0, AMEVCNTVOFF07_EL2
mrs x0, AMEVCNTVOFF08_EL2
mrs x0, AMEVCNTVOFF09_EL2
mrs x0, AMEVCNTVOFF010_EL2
mrs x0, AMEVCNTVOFF011_EL2
mrs x0, AMEVCNTVOFF012_EL2
mrs x0, AMEVCNTVOFF013_EL2
mrs x0, AMEVCNTVOFF014_EL2
mrs x0, AMEVCNTVOFF015_EL2
msr AMEVCNTVOFF10_EL2, x0
msr AMEVCNTVOFF11_EL2, x0
msr AMEVCNTVOFF12_EL2, x0
msr AMEVCNTVOFF13_EL2, x0
msr AMEVCNTVOFF14_EL2, x0
msr AMEVCNTVOFF15_EL2, x0
msr AMEVCNTVOFF16_EL2, x0
msr AMEVCNTVOFF17_EL2, x0
msr AMEVCNTVOFF18_EL2, x0
msr AMEVCNTVOFF19_EL2, x0
msr AMEVCNTVOFF110_EL2, x0
msr AMEVCNTVOFF111_EL2, x0
msr AMEVCNTVOFF112_EL2, x0
msr AMEVCNTVOFF113_EL2, x0
msr AMEVCNTVOFF114_EL2, x0
msr AMEVCNTVOFF115_EL2, x0
mrs x0, AMEVCNTVOFF10_EL2
mrs x0, AMEVCNTVOFF11_EL2
mrs x0, AMEVCNTVOFF12_EL2
mrs x0, AMEVCNTVOFF13_EL2
mrs x0, AMEVCNTVOFF14_EL2
mrs x0, AMEVCNTVOFF15_EL2
mrs x0, AMEVCNTVOFF16_EL2
mrs x0, AMEVCNTVOFF17_EL2
mrs x0, AMEVCNTVOFF18_EL2
mrs x0, AMEVCNTVOFF19_EL2
mrs x0, AMEVCNTVOFF110_EL2
mrs x0, AMEVCNTVOFF111_EL2
mrs x0, AMEVCNTVOFF112_EL2
mrs x0, AMEVCNTVOFF113_EL2
mrs x0, AMEVCNTVOFF114_EL2
mrs x0, AMEVCNTVOFF115_EL2

// CHECK:  .text
// CHECK-NEXT:  mrs     x0, AMCG1IDR_EL0        // encoding: [0xc0,0xd2,0x3b,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF00_EL2, x0   // encoding: [0x00,0xd8,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF01_EL2, x0   // encoding: [0x20,0xd8,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF02_EL2, x0   // encoding: [0x40,0xd8,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF03_EL2, x0   // encoding: [0x60,0xd8,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF04_EL2, x0   // encoding: [0x80,0xd8,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF05_EL2, x0   // encoding: [0xa0,0xd8,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF06_EL2, x0   // encoding: [0xc0,0xd8,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF07_EL2, x0   // encoding: [0xe0,0xd8,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF08_EL2, x0   // encoding: [0x00,0xd9,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF09_EL2, x0   // encoding: [0x20,0xd9,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF010_EL2, x0  // encoding: [0x40,0xd9,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF011_EL2, x0  // encoding: [0x60,0xd9,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF012_EL2, x0  // encoding: [0x80,0xd9,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF013_EL2, x0  // encoding: [0xa0,0xd9,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF014_EL2, x0  // encoding: [0xc0,0xd9,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF015_EL2, x0  // encoding: [0xe0,0xd9,0x1c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF00_EL2   // encoding: [0x00,0xd8,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF01_EL2   // encoding: [0x20,0xd8,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF02_EL2   // encoding: [0x40,0xd8,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF03_EL2   // encoding: [0x60,0xd8,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF04_EL2   // encoding: [0x80,0xd8,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF05_EL2   // encoding: [0xa0,0xd8,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF06_EL2   // encoding: [0xc0,0xd8,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF07_EL2   // encoding: [0xe0,0xd8,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF08_EL2   // encoding: [0x00,0xd9,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF09_EL2   // encoding: [0x20,0xd9,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF010_EL2  // encoding: [0x40,0xd9,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF011_EL2  // encoding: [0x60,0xd9,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF012_EL2  // encoding: [0x80,0xd9,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF013_EL2  // encoding: [0xa0,0xd9,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF014_EL2  // encoding: [0xc0,0xd9,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF015_EL2  // encoding: [0xe0,0xd9,0x3c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF10_EL2, x0   // encoding: [0x00,0xda,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF11_EL2, x0   // encoding: [0x20,0xda,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF12_EL2, x0   // encoding: [0x40,0xda,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF13_EL2, x0   // encoding: [0x60,0xda,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF14_EL2, x0   // encoding: [0x80,0xda,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF15_EL2, x0   // encoding: [0xa0,0xda,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF16_EL2, x0   // encoding: [0xc0,0xda,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF17_EL2, x0   // encoding: [0xe0,0xda,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF18_EL2, x0   // encoding: [0x00,0xdb,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF19_EL2, x0   // encoding: [0x20,0xdb,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF110_EL2, x0  // encoding: [0x40,0xdb,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF111_EL2, x0  // encoding: [0x60,0xdb,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF112_EL2, x0  // encoding: [0x80,0xdb,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF113_EL2, x0  // encoding: [0xa0,0xdb,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF114_EL2, x0  // encoding: [0xc0,0xdb,0x1c,0xd5]
// CHECK-NEXT:  msr     AMEVCNTVOFF115_EL2, x0  // encoding: [0xe0,0xdb,0x1c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF10_EL2   // encoding: [0x00,0xda,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF11_EL2   // encoding: [0x20,0xda,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF12_EL2   // encoding: [0x40,0xda,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF13_EL2   // encoding: [0x60,0xda,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF14_EL2   // encoding: [0x80,0xda,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF15_EL2   // encoding: [0xa0,0xda,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF16_EL2   // encoding: [0xc0,0xda,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF17_EL2   // encoding: [0xe0,0xda,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF18_EL2   // encoding: [0x00,0xdb,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF19_EL2   // encoding: [0x20,0xdb,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF110_EL2  // encoding: [0x40,0xdb,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF111_EL2  // encoding: [0x60,0xdb,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF112_EL2  // encoding: [0x80,0xdb,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF113_EL2  // encoding: [0xa0,0xdb,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF114_EL2  // encoding: [0xc0,0xdb,0x3c,0xd5]
// CHECK-NEXT:  mrs     x0, AMEVCNTVOFF115_EL2  // encoding: [0xe0,0xdb,0x3c,0xd5]


// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF00_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF01_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF02_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF03_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF04_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF05_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF06_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF07_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF08_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF09_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF010_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF011_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF012_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF013_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF014_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF015_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF00_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF01_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF02_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF03_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF04_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF05_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF06_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF07_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF08_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF09_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF010_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF011_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF012_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF013_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF014_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF015_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF10_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF11_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF12_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF13_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF14_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF15_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF16_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF17_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF18_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF19_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF110_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF111_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF112_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF113_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF114_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT: msr AMEVCNTVOFF115_EL2, x0
// CHECK-ERROR-NEXT:     ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF10_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF11_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF12_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF13_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF14_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF15_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF16_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF17_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF18_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF19_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF110_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF111_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF112_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF113_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF114_EL2
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT: mrs x0, AMEVCNTVOFF115_EL2
// CHECK-ERROR-NEXT:         ^
