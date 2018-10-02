// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+mte   < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOMTE
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-mte   < %s 2>&1 | FileCheck %s --check-prefix=NOMTE

dc igvac, x0
dc igsw, x1
dc cgsw, x2
dc cigsw, x3
dc cgvac, x4
dc cgvap, x5
dc cgvadp, x6
dc cigvac, x7
dc gva, x8
dc igdvac, x9
dc igdsw, x10
dc cgdsw, x11
dc cigdsw, x12
dc cgdvac, x13
dc cgdvap, x14
dc cgdvadp, x15
dc cigdvac, x16
dc gzva, x17

// CHECK: dc igvac, x0          // encoding: [0x60,0x76,0x08,0xd5]
// CHECK: dc igsw, x1           // encoding: [0x81,0x76,0x08,0xd5]
// CHECK: dc cgsw, x2           // encoding: [0x82,0x7a,0x08,0xd5]
// CHECK: dc cigsw, x3          // encoding: [0x83,0x7e,0x08,0xd5]
// CHECK: dc cgvac, x4          // encoding: [0x64,0x7a,0x0b,0xd5]
// CHECK: dc cgvap, x5          // encoding: [0x65,0x7c,0x0b,0xd5]
// CHECK: dc cgvadp, x6         // encoding: [0x66,0x7d,0x0b,0xd5]
// CHECK: dc cigvac, x7         // encoding: [0x67,0x7e,0x0b,0xd5]
// CHECK: dc gva, x8            // encoding: [0x68,0x74,0x0b,0xd5]
// CHECK: dc igdvac, x9         // encoding: [0xa9,0x76,0x08,0xd5]
// CHECK: dc igdsw, x10         // encoding: [0xca,0x76,0x08,0xd5]
// CHECK: dc cgdsw, x11         // encoding: [0xcb,0x7a,0x08,0xd5]
// CHECK: dc cigdsw, x12        // encoding: [0xcc,0x7e,0x08,0xd5]
// CHECK: dc cgdvac, x13        // encoding: [0xad,0x7a,0x0b,0xd5]
// CHECK: dc cgdvap, x14        // encoding: [0xae,0x7c,0x0b,0xd5]
// CHECK: dc cgdvadp, x15       // encoding: [0xaf,0x7d,0x0b,0xd5]
// CHECK: dc cigdvac, x16       // encoding: [0xb0,0x7e,0x0b,0xd5]
// CHECK: dc gzva, x17          // encoding: [0x91,0x74,0x0b,0xd5]

// NOMTE: DC IGVAC requires mte
// NOMTE: DC IGSW requires mte
// NOMTE: DC CGSW requires mte
// NOMTE: DC CIGSW requires mte
// NOMTE: DC CGVAC requires mte
// NOMTE: DC CGVAP requires mte
// NOMTE: DC CGVADP requires mte
// NOMTE: DC CIGVAC requires mte
// NOMTE: DC GVA requires mte
// NOMTE: DC IGDVAC requires mte
// NOMTE: DC IGDSW requires mte
// NOMTE: DC CGDSW requires mte
// NOMTE: DC CIGDSW requires mte
// NOMTE: DC CGDVAC requires mte
// NOMTE: DC CGDVAP requires mte
// NOMTE: DC CGDVADP requires mte
// NOMTE: DC CIGDVAC requires mte
// NOMTE: DC GZVA requires mte

mrs x0, tco
mrs x1, gcr_el1
mrs x2, rgsr_el1
mrs x3, tfsr_el1
mrs x4, tfsr_el2
mrs x5, tfsr_el3
mrs x6, tfsr_el12
mrs x7, tfsre0_el1

// CHECK: mrs x0, TCO           // encoding: [0xe0,0x42,0x3b,0xd5]
// CHECK: mrs x1, GCR_EL1       // encoding: [0xc1,0x10,0x38,0xd5]
// CHECK: mrs x2, RGSR_EL1      // encoding: [0xa2,0x10,0x38,0xd5]
// CHECK: mrs x3, TFSR_EL1      // encoding: [0x03,0x65,0x38,0xd5]
// CHECK: mrs x4, TFSR_EL2      // encoding: [0x04,0x65,0x3c,0xd5]
// CHECK: mrs x5, TFSR_EL3      // encoding: [0x05,0x66,0x3e,0xd5]
// CHECK: mrs x6, TFSR_EL12     // encoding: [0x06,0x66,0x3d,0xd5]
// CHECK: mrs x7, TFSRE0_EL1    // encoding: [0x27,0x66,0x38,0xd5]

// NOMTE: expected readable system register
// NOMTE-NEXT: tco
// NOMTE: expected readable system register
// NOMTE-NEXT: gcr_el1
// NOMTE: expected readable system register
// NOMTE-NEXT: rgsr_el1
// NOMTE: expected readable system register
// NOMTE-NEXT: tfsr_el1
// NOMTE: expected readable system register
// NOMTE-NEXT: tfsr_el2
// NOMTE: expected readable system register
// NOMTE-NEXT: tfsr_el3
// NOMTE: expected readable system register
// NOMTE-NEXT: tfsr_el12
// NOMTE: expected readable system register
// NOMTE-NEXT: tfsre0_el1

msr tco, #0

// CHECK: msr TCO, #0           // encoding: [0x9f,0x40,0x03,0xd5]

// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: tco

msr tco, x0
msr gcr_el1, x1
msr rgsr_el1, x2
msr tfsr_el1, x3
msr tfsr_el2, x4
msr tfsr_el3, x5
msr tfsr_el12, x6
msr tfsre0_el1, x7

// CHECK: msr TCO, x0           // encoding: [0xe0,0x42,0x1b,0xd5]
// CHECK: msr GCR_EL1, x1       // encoding: [0xc1,0x10,0x18,0xd5]
// CHECK: msr RGSR_EL1, x2      // encoding: [0xa2,0x10,0x18,0xd5]
// CHECK: msr TFSR_EL1, x3      // encoding: [0x03,0x65,0x18,0xd5]
// CHECK: msr TFSR_EL2, x4      // encoding: [0x04,0x65,0x1c,0xd5]
// CHECK: msr TFSR_EL3, x5      // encoding: [0x05,0x66,0x1e,0xd5]
// CHECK: msr TFSR_EL12, x6     // encoding: [0x06,0x66,0x1d,0xd5]
// CHECK: msr TFSRE0_EL1, x7    // encoding: [0x27,0x66,0x18,0xd5]

// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: tco
// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: gcr_el1
// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: rgsr_el1
// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: tfsr_el1
// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: tfsr_el2
// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: tfsr_el3
// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: tfsr_el12
// NOMTE: expected writable system register or pstate
// NOMTE-NEXT: tfsre0_el1
