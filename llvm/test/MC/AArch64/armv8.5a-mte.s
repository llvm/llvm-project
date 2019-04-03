// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+mte   < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOMTE
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-mte   < %s 2>&1 | FileCheck %s --check-prefix=NOMTE

irg x0, x1
irg sp, x1
irg x0, sp
irg x0, x1, x2
irg sp, x1, x2

// CHECK: irg x0, x1            // encoding: [0x20,0x10,0xdf,0x9a]
// CHECK: irg sp, x1            // encoding: [0x3f,0x10,0xdf,0x9a]
// CHECK: irg x0, sp            // encoding: [0xe0,0x13,0xdf,0x9a]
// CHECK: irg x0, x1, x2        // encoding: [0x20,0x10,0xc2,0x9a]
// CHECK: irg sp, x1, x2        // encoding: [0x3f,0x10,0xc2,0x9a]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: irg x0, x1
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: irg sp, x1
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: irg x0, sp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: irg x0, x1, x2
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: irg sp, x1, x2

addg x0, x1, #0, #1
addg sp, x2, #32, #3
addg x0, sp, #64, #5
addg x3, x4, #1008, #6
addg x5, x6, #112, #15

subg x0, x1, #0, #1
subg sp, x2, #32, #3
subg x0, sp, #64, #5
subg x3, x4, #1008, #6
subg x5, x6, #112, #15

// CHECK: addg x0, x1, #0, #1   // encoding: [0x20,0x04,0x80,0x91]
// CHECK: addg sp, x2, #32, #3   // encoding: [0x5f,0x0c,0x82,0x91]
// CHECK: addg x0, sp, #64, #5   // encoding: [0xe0,0x17,0x84,0x91]
// CHECK: addg x3, x4, #1008, #6  // encoding: [0x83,0x18,0xbf,0x91]
// CHECK: addg x5, x6, #112, #15  // encoding: [0xc5,0x3c,0x87,0x91]

// CHECK: subg x0, x1, #0, #1   // encoding: [0x20,0x04,0x80,0xd1]
// CHECK: subg sp, x2, #32, #3   // encoding: [0x5f,0x0c,0x82,0xd1]
// CHECK: subg x0, sp, #64, #5   // encoding: [0xe0,0x17,0x84,0xd1]
// CHECK: subg x3, x4, #1008, #6  // encoding: [0x83,0x18,0xbf,0xd1]
// CHECK: subg x5, x6, #112, #15  // encoding: [0xc5,0x3c,0x87,0xd1]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: addg x0, x1, #0, #1
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: addg sp, x2, #32, #3
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: addg x0, sp, #64, #5
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: addg x3, x4, #1008, #6
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: addg x5, x6, #112, #15

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: subg x0, x1, #0, #1
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: subg sp, x2, #32, #3
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: subg x0, sp, #64, #5
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: subg x3, x4, #1008, #6
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: subg x5, x6, #112, #15

gmi x0, x1, x2
gmi x3, sp, x4
gmi xzr, x0, x30
gmi x30, x0, xzr

// CHECK: gmi x0, x1, x2        // encoding: [0x20,0x14,0xc2,0x9a]
// CHECK: gmi x3, sp, x4        // encoding: [0xe3,0x17,0xc4,0x9a]
// CHECK: gmi xzr, x0, x30      // encoding: [0x1f,0x14,0xde,0x9a]
// CHECK: gmi x30, x0, xzr      // encoding: [0x1e,0x14,0xdf,0x9a]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: gmi x0, x1, x2
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: gmi x3, sp, x4
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: gmi xzr, x0, x30
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: gmi x30, x0, xzr

stg x0,  [x1]
stg x1,  [x1, #-4096]
stg x2,  [x2, #4080]
stg x3,  [sp, #16]
stg sp,  [sp, #16]

// CHECK: stg x0,  [x1]              // encoding: [0x20,0x08,0x20,0xd9]
// CHECK: stg x1,  [x1, #-4096]      // encoding: [0x21,0x08,0x30,0xd9]
// CHECK: stg x2,  [x2, #4080]       // encoding: [0x42,0xf8,0x2f,0xd9]
// CHECK: stg x3,  [sp, #16]         // encoding: [0xe3,0x1b,0x20,0xd9]
// CHECK: stg sp,  [sp, #16]         // encoding: [0xff,0x1b,0x20,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg

stzg x0,  [x1]
stzg x1,  [x1, #-4096]
stzg x2,  [x2, #4080]
stzg x3,  [sp, #16]
stzg sp,  [sp, #16]

// CHECK: stzg x0,  [x1]             // encoding: [0x20,0x08,0x60,0xd9]
// CHECK: stzg x1,  [x1, #-4096]     // encoding: [0x21,0x08,0x70,0xd9]
// CHECK: stzg x2,  [x2, #4080]      // encoding: [0x42,0xf8,0x6f,0xd9]
// CHECK: stzg x3,  [sp, #16]        // encoding: [0xe3,0x1b,0x60,0xd9]
// CHECK: stzg sp,  [sp, #16]        // encoding: [0xff,0x1b,0x60,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg

stg x0,  [x1, #-4096]!
stg x1,  [x2, #4080]!
stg x2,  [sp, #16]!
stg sp,  [sp, #16]!

// CHECK: stg x0,  [x1, #-4096]!      // encoding: [0x20,0x0c,0x30,0xd9]
// CHECK: stg x1,  [x2, #4080]!       // encoding: [0x41,0xfc,0x2f,0xd9]
// CHECK: stg x2,  [sp, #16]!         // encoding: [0xe2,0x1f,0x20,0xd9]
// CHECK: stg sp,  [sp, #16]!         // encoding: [0xff,0x1f,0x20,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg

stzg x0,  [x1, #-4096]!
stzg x1,  [x2, #4080]!
stzg x2,  [sp, #16]!
stzg sp,  [sp, #16]!

// CHECK: stzg x0,  [x1, #-4096]!     // encoding: [0x20,0x0c,0x70,0xd9]
// CHECK: stzg x1,  [x2, #4080]!      // encoding: [0x41,0xfc,0x6f,0xd9]
// CHECK: stzg x2,  [sp, #16]!        // encoding: [0xe2,0x1f,0x60,0xd9]
// CHECK: stzg sp,  [sp, #16]!        // encoding: [0xff,0x1f,0x60,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg

stg x0,  [x1], #-4096
stg x1,  [x2], #4080
stg x2,  [sp], #16
stg sp,  [sp], #16

// CHECK: stg x0,  [x1], #-4096       // encoding: [0x20,0x04,0x30,0xd9]
// CHECK: stg x1,  [x2], #4080        // encoding: [0x41,0xf4,0x2f,0xd9]
// CHECK: stg x2,  [sp], #16          // encoding: [0xe2,0x17,0x20,0xd9]
// CHECK: stg sp,  [sp], #16          // encoding: [0xff,0x17,0x20,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stg

stzg x0,  [x1], #-4096
stzg x1,  [x2], #4080
stzg x2,  [sp], #16
stzg sp,  [sp], #16

// CHECK: stzg x0,  [x1], #-4096      // encoding: [0x20,0x04,0x70,0xd9]
// CHECK: stzg x1,  [x2], #4080       // encoding: [0x41,0xf4,0x6f,0xd9]
// CHECK: stzg x2,  [sp], #16         // encoding: [0xe2,0x17,0x60,0xd9]
// CHECK: stzg sp,  [sp], #16         // encoding: [0xff,0x17,0x60,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stzg

st2g x0,  [x1]
st2g x1,  [x1, #-4096]
st2g x2,  [x2, #4080]
st2g x3,  [sp, #16]
st2g sp,  [sp, #16]

// CHECK: st2g x0,  [x1]              // encoding: [0x20,0x08,0xa0,0xd9]
// CHECK: st2g x1,  [x1, #-4096]      // encoding: [0x21,0x08,0xb0,0xd9]
// CHECK: st2g x2,  [x2, #4080]       // encoding: [0x42,0xf8,0xaf,0xd9]
// CHECK: st2g x3,  [sp, #16]         // encoding: [0xe3,0x1b,0xa0,0xd9]
// CHECK: st2g sp,  [sp, #16]         // encoding: [0xff,0x1b,0xa0,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g

stz2g x0,  [x1]
stz2g x1,  [x1, #-4096]
stz2g x2,  [x2, #4080]
stz2g x3,  [sp, #16]
stz2g sp,  [sp, #16]

// CHECK: stz2g x0,  [x1]             // encoding: [0x20,0x08,0xe0,0xd9]
// CHECK: stz2g x1,  [x1, #-4096]     // encoding: [0x21,0x08,0xf0,0xd9]
// CHECK: stz2g x2,  [x2, #4080]      // encoding: [0x42,0xf8,0xef,0xd9]
// CHECK: stz2g x3,  [sp, #16]        // encoding: [0xe3,0x1b,0xe0,0xd9]
// CHECK: stz2g sp,  [sp, #16]        // encoding: [0xff,0x1b,0xe0,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g

st2g x0,  [x1, #-4096]!
st2g x1,  [x2, #4080]!
st2g x2,  [sp, #16]!
st2g sp,  [sp, #16]!

// CHECK: st2g x0,  [x1, #-4096]!     // encoding: [0x20,0x0c,0xb0,0xd9]
// CHECK: st2g x1,  [x2, #4080]!      // encoding: [0x41,0xfc,0xaf,0xd9]
// CHECK: st2g x2,  [sp, #16]!        // encoding: [0xe2,0x1f,0xa0,0xd9]
// CHECK: st2g sp,  [sp, #16]!        // encoding: [0xff,0x1f,0xa0,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g

stz2g x0,  [x1, #-4096]!
stz2g x1,  [x2, #4080]!
stz2g x2,  [sp, #16]!
stz2g sp,  [sp, #16]!

// CHECK: stz2g x0,  [x1, #-4096]!    // encoding: [0x20,0x0c,0xf0,0xd9]
// CHECK: stz2g x1,  [x2, #4080]!     // encoding: [0x41,0xfc,0xef,0xd9]
// CHECK: stz2g x2,  [sp, #16]!       // encoding: [0xe2,0x1f,0xe0,0xd9]
// CHECK: stz2g sp,  [sp, #16]!       // encoding: [0xff,0x1f,0xe0,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g

st2g x0,  [x1], #-4096
st2g x1,  [x2], #4080
st2g x2,  [sp], #16
st2g sp,  [sp], #16

// CHECK: st2g x0,  [x1], #-4096      // encoding: [0x20,0x04,0xb0,0xd9]
// CHECK: st2g x1,  [x2], #4080       // encoding: [0x41,0xf4,0xaf,0xd9]
// CHECK: st2g x2,  [sp], #16         // encoding: [0xe2,0x17,0xa0,0xd9]
// CHECK: st2g sp,  [sp], #16         // encoding: [0xff,0x17,0xa0,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: st2g

stz2g x0,  [x1], #-4096
stz2g x1,  [x2], #4080
stz2g x2,  [sp], #16
stz2g sp,  [sp], #16

// CHECK: stz2g x0,  [x1], #-4096     // encoding: [0x20,0x04,0xf0,0xd9]
// CHECK: stz2g x1,  [x2], #4080      // encoding: [0x41,0xf4,0xef,0xd9]
// CHECK: stz2g x2,  [sp], #16        // encoding: [0xe2,0x17,0xe0,0xd9]
// CHECK: stz2g sp,  [sp], #16        // encoding: [0xff,0x17,0xe0,0xd9]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stz2g

stgp x0, x1, [x2]
stgp x0, x1, [x2, #-1024]
stgp x0, x1, [x2, #1008]
stgp x0, x1, [sp, #16]
stgp xzr, x1, [x2, #16]
stgp x0, xzr, [x2, #16]

// CHECK: stgp x0, x1, [x2]           // encoding: [0x40,0x04,0x00,0x69]
// CHECK: stgp x0, x1, [x2, #-1024]   // encoding: [0x40,0x04,0x20,0x69]
// CHECK: stgp x0, x1, [x2, #1008]    // encoding: [0x40,0x84,0x1f,0x69]
// CHECK: stgp x0, x1, [sp, #16]      // encoding: [0xe0,0x87,0x00,0x69]
// CHECK: stgp xzr, x1, [x2, #16]     // encoding: [0x5f,0x84,0x00,0x69]
// CHECK: stgp x0, xzr, [x2, #16]     // encoding: [0x40,0xfc,0x00,0x69]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp

stgp x0, x1, [x2, #-1024]!
stgp x0, x1, [x2, #1008]!
stgp x0, x1, [sp, #16]!
stgp xzr, x1, [x2, #16]!
stgp x0, xzr, [x2, #16]!

// CHECK: stgp x0, x1, [x2, #-1024]!   // encoding: [0x40,0x04,0xa0,0x69]
// CHECK: stgp x0, x1, [x2, #1008]!    // encoding: [0x40,0x84,0x9f,0x69]
// CHECK: stgp x0, x1, [sp, #16]!      // encoding: [0xe0,0x87,0x80,0x69]
// CHECK: stgp xzr, x1, [x2, #16]!     // encoding: [0x5f,0x84,0x80,0x69]
// CHECK: stgp x0, xzr, [x2, #16]!     // encoding: [0x40,0xfc,0x80,0x69]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp

stgp x0, x1, [x2], #-1024
stgp x0, x1, [x2], #1008
stgp x0, x1, [sp], #16
stgp xzr, x1, [x2], #16
stgp x0, xzr, [x2], #16

// CHECK: stgp x0, x1, [x2], #-1024    // encoding: [0x40,0x04,0xa0,0x68]
// CHECK: stgp x0, x1, [x2], #1008     // encoding: [0x40,0x84,0x9f,0x68]
// CHECK: stgp x0, x1, [sp], #16       // encoding: [0xe0,0x87,0x80,0x68]
// CHECK: stgp xzr, x1, [x2], #16      // encoding: [0x5f,0x84,0x80,0x68]
// CHECK: stgp x0, xzr, [x2], #16      // encoding: [0x40,0xfc,0x80,0x68]

// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp
// NOMTE:      instruction requires: mte
// NOMTE-NEXT: stgp

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
mrs x7, gmid_el1

// CHECK: mrs x0, TCO           // encoding: [0xe0,0x42,0x3b,0xd5]
// CHECK: mrs x1, GCR_EL1       // encoding: [0xc1,0x10,0x38,0xd5]
// CHECK: mrs x2, RGSR_EL1      // encoding: [0xa2,0x10,0x38,0xd5]
// CHECK: mrs x3, TFSR_EL1      // encoding: [0x03,0x65,0x38,0xd5]
// CHECK: mrs x4, TFSR_EL2      // encoding: [0x04,0x65,0x3c,0xd5]
// CHECK: mrs x5, TFSR_EL3      // encoding: [0x05,0x66,0x3e,0xd5]
// CHECK: mrs x6, TFSR_EL12     // encoding: [0x06,0x66,0x3d,0xd5]
// CHECK: mrs x7, TFSRE0_EL1    // encoding: [0x27,0x66,0x38,0xd5]
// CHECK: mrs x7, GMID_EL1      // encoding: [0x87,0x00,0x39,0xd5]

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
// NOMTE: expected readable system register
// NOMTE-NEXT: gmid_el1

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

subp  x0, x1, x2
subp  x0, sp, sp
subps x0, x1, x2
subps x0, sp, sp

// CHECK: subp  x0, x1, x2  // encoding: [0x20,0x00,0xc2,0x9a]
// CHECK: subp  x0, sp, sp  // encoding: [0xe0,0x03,0xdf,0x9a]
// CHECK: subps x0, x1, x2  // encoding: [0x20,0x00,0xc2,0xba]
// CHECK: subps x0, sp, sp  // encoding: [0xe0,0x03,0xdf,0xba]

// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte

subps xzr, x0, x1
cmpp  x0, x1
subps xzr, sp, sp
cmpp  sp, sp

// CHECK: subps xzr, x0, x1 // encoding: [0x1f,0x00,0xc1,0xba]
// CHECK: subps xzr, x0, x1 // encoding: [0x1f,0x00,0xc1,0xba]
// CHECK: subps xzr, sp, sp // encoding: [0xff,0x03,0xdf,0xba]
// CHECK: subps xzr, sp, sp // encoding: [0xff,0x03,0xdf,0xba]

// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte

ldg X0, [X1, #0]
ldg X2, [sp, #-4096]
ldg x3, [x4, #4080]

// CHECK: ldg x0, [x1]         // encoding: [0x20,0x00,0x60,0xd9]
// CHECK: ldg x2, [sp, #-4096] // encoding: [0xe2,0x03,0x70,0xd9]
// CHECK: ldg x3, [x4, #4080]  // encoding: [0x83,0xf0,0x6f,0xd9]

// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte

ldgm x0, [x1]
ldgm x1, [sp]
ldgm xzr, [x2]

// CHECK: ldgm x0, [x1]  // encoding: [0x20,0x00,0xe0,0xd9]
// CHECK: ldgm x1, [sp]  // encoding: [0xe1,0x03,0xe0,0xd9]
// CHECK: ldgm xzr, [x2] // encoding: [0x5f,0x00,0xe0,0xd9]

// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte

stgm x0, [x1]
stgm x1, [sp]
stgm xzr, [x2]

// CHECK: stgm x0, [x1]  // encoding: [0x20,0x00,0xa0,0xd9]
// CHECK: stgm x1, [sp]  // encoding: [0xe1,0x03,0xa0,0xd9]
// CHECK: stgm xzr, [x2] // encoding: [0x5f,0x00,0xa0,0xd9]

// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte

stzgm x0, [x1]
stzgm x1, [sp]
stzgm xzr, [x2]

// CHECK: stzgm x0, [x1]  // encoding: [0x20,0x00,0x20,0xd9]
// CHECK: stzgm x1, [sp]  // encoding: [0xe1,0x03,0x20,0xd9]
// CHECK: stzgm xzr, [x2] // encoding: [0x5f,0x00,0x20,0xd9]

// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte
// NOMTE: instruction requires: mte
