// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s | FileCheck %s --check-prefix=CHECK
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-V84

//------------------------------------------------------------------------------
// Armv8.4-A LDAPR and STLR instructions with immediate offsets
//------------------------------------------------------------------------------

STLURB   WZR, [X10]
STLURB   W1, [X10]
STLURB   W1, [X10, #-256]
stlurb   w2, [x11, #255]
STLURB   W3, [SP, #-3]

//CHECK:       stlurb  w1, [x10]               // encoding: [0x41,0x01,0x00,0x19]
//CHECK-NEXT:  stlurb  w1, [x10, #-256]        // encoding: [0x41,0x01,0x10,0x19]
//CHECK-NEXT:  stlurb  w2, [x11, #255]         // encoding: [0x62,0xf1,0x0f,0x19]
//CHECK-NEXT:  stlurb  w3, [sp, #-3]           // encoding: [0xe3,0xd3,0x1f,0x19]

ldapurb  wzr, [x12]
ldapurb  w4, [x12]
ldapurb  w4, [x12, #-256]
LDAPURB  W5, [X13, #255]
LDAPURB  W6, [SP, #-2]

//CHECK:       ldapurb wzr, [x12]              // encoding: [0x9f,0x01,0x40,0x19]
//CHECK-NEXT:  ldapurb w4, [x12]               // encoding: [0x84,0x01,0x40,0x19]
//CHECK-NEXT:  ldapurb w4, [x12, #-256]        // encoding: [0x84,0x01,0x50,0x19]
//CHECK-NEXT:  ldapurb w5, [x13, #255]         // encoding: [0xa5,0xf1,0x4f,0x19]
//CHECK-NEXT:  ldapurb w6, [sp, #-2]           // encoding: [0xe6,0xe3,0x5f,0x19]

LDAPURSB W7, [X14]
LDAPURSB W7, [X14, #-256]
ldapursb w8, [x15, #255]
ldapursb w9, [sp, #-1]

//CHECK:       ldapursb    w7, [x14]           // encoding: [0xc7,0x01,0xc0,0x19]
//CHECK-NEXT:  ldapursb  w7, [x14, #-256]      // encoding: [0xc7,0x01,0xd0,0x19]
//CHECK-NEXT:  ldapursb  w8, [x15, #255]       // encoding: [0xe8,0xf1,0xcf,0x19]
//CHECK-NEXT:  ldapursb  w9, [sp, #-1]         // encoding: [0xe9,0xf3,0xdf,0x19]

LDAPURSB X0, [X16]
LDAPURSB X0, [X16, #-256]
LDAPURSB X1, [X17, #255]
ldapursb x2, [sp, #0]
ldapursb x2, [sp]

//CHECK:       ldapursb    x0, [x16]           // encoding: [0x00,0x02,0x80,0x19]
//CHECK-NEXT:  ldapursb  x0, [x16, #-256]      // encoding: [0x00,0x02,0x90,0x19]
//CHECK-NEXT:  ldapursb  x1, [x17, #255]       // encoding: [0x21,0xf2,0x8f,0x19]
//CHECK-NEXT:  ldapursb  x2, [sp]              // encoding: [0xe2,0x03,0x80,0x19]
//CHECK-NEXT:  ldapursb  x2, [sp]              // encoding: [0xe2,0x03,0x80,0x19]

stlurh   w10, [x18]
stlurh   w10, [x18, #-256]
STLURH   W11, [X19, #255]
STLURH   W12, [SP, #1]

//CHECK:       stlurh    w10, [x18]            // encoding: [0x4a,0x02,0x00,0x59]
//CHECK-NEXT:  stlurh  w10, [x18, #-256]       // encoding: [0x4a,0x02,0x10,0x59]
//CHECK-NEXT:  stlurh  w11, [x19, #255]        // encoding: [0x6b,0xf2,0x0f,0x59]
//CHECK-NEXT:  stlurh  w12, [sp, #1]           // encoding: [0xec,0x13,0x00,0x59]

LDAPURH  W13, [X20]
LDAPURH  W13, [X20, #-256]
ldapurh  w14, [x21, #255]
LDAPURH  W15, [SP, #2]

//CHECK:       ldapurh   w13, [x20]            // encoding: [0x8d,0x02,0x40,0x59]
//CHECK-NEXT:  ldapurh w13, [x20, #-256]       // encoding: [0x8d,0x02,0x50,0x59]
//CHECK-NEXT:  ldapurh w14, [x21, #255]        // encoding: [0xae,0xf2,0x4f,0x59]
//CHECK-NEXT:  ldapurh w15, [sp, #2]           // encoding: [0xef,0x23,0x40,0x59]

LDAPURSH W16, [X22]
LDAPURSH W16, [X22, #-256]
LDAPURSH W17, [X23, #255]
ldapursh w18, [sp, #3]

//CHECK:       ldapursh    w16, [x22]          // encoding: [0xd0,0x02,0xc0,0x59]
//CHECK-NEXT:  ldapursh  w16, [x22, #-256]     // encoding: [0xd0,0x02,0xd0,0x59]
//CHECK-NEXT:  ldapursh  w17, [x23, #255]      // encoding: [0xf1,0xf2,0xcf,0x59]
//CHECK-NEXT:  ldapursh  w18, [sp, #3]         // encoding: [0xf2,0x33,0xc0,0x59]

ldapursh x3, [x24]
ldapursh x3, [x24, #-256]
LDAPURSH X4, [X25, #255]
LDAPURSH X5, [SP, #4]

//CHECK:       ldapursh    x3, [x24]          // encoding: [0x03,0x03,0x80,0x59]
//CHECK-NEXT:  ldapursh  x3, [x24, #-256]     // encoding: [0x03,0x03,0x90,0x59]
//CHECK-NEXT:  ldapursh  x4, [x25, #255]      // encoding: [0x24,0xf3,0x8f,0x59]
//CHECK-NEXT:  ldapursh  x5, [sp, #4]         // encoding: [0xe5,0x43,0x80,0x59]

STLUR    W19, [X26]
STLUR    W19, [X26, #-256]
stlur    w20, [x27, #255]
STLUR    W21, [SP, #5]

//CHECK:       stlur   w19, [x26]            // encoding: [0x53,0x03,0x00,0x99]
//CHECK-NEXT:  stlur w19, [x26, #-256]       // encoding: [0x53,0x03,0x10,0x99]
//CHECK-NEXT:  stlur w20, [x27, #255]        // encoding: [0x74,0xf3,0x0f,0x99]
//CHECK-NEXT:  stlur w21, [sp, #5]           // encoding: [0xf5,0x53,0x00,0x99]

LDAPUR   W22, [X28]
LDAPUR   W22, [X28, #-256]
LDAPUR   W23, [X29, #255]
ldapur   w24, [sp, #6]

//CHECK:       ldapur    w22, [x28]          // encoding: [0x96,0x03,0x40,0x99]
//CHECK-NEXT:  ldapur  w22, [x28, #-256]     // encoding: [0x96,0x03,0x50,0x99]
//CHECK-NEXT:  ldapur  w23, [x29, #255]      // encoding: [0xb7,0xf3,0x4f,0x99]
//CHECK-NEXT:  ldapur  w24, [sp, #6]         // encoding: [0xf8,0x63,0x40,0x99]

ldapursw x6, [x30]
ldapursw x6, [x30, #-256]
LDAPURSW X7, [X0, #255]
LDAPURSW X8, [SP, #7]

//CHECK:       ldapursw    x6, [x30]         // encoding: [0xc6,0x03,0x80,0x99]
//CHECK-NEXT:  ldapursw  x6, [x30, #-256]    // encoding: [0xc6,0x03,0x90,0x99]
//CHECK-NEXT:  ldapursw  x7, [x0, #255]      // encoding: [0x07,0xf0,0x8f,0x99]
//CHECK-NEXT:  ldapursw  x8, [sp, #7]        // encoding: [0xe8,0x73,0x80,0x99]

STLUR    X9, [X1]
STLUR    X9, [X1, #-256]
stlur    x10, [x2, #255]
STLUR    X11, [SP, #8]

//CHECK:       stlur   x9, [x1]              // encoding: [0x29,0x00,0x00,0xd9]
//CHECK-NEXT:  stlur x9, [x1, #-256]         // encoding: [0x29,0x00,0x10,0xd9]
//CHECK-NEXT:  stlur x10, [x2, #255]         // encoding: [0x4a,0xf0,0x0f,0xd9]
//CHECK-NEXT:  stlur x11, [sp, #8]           // encoding: [0xeb,0x83,0x00,0xd9]

LDAPUR   X12, [X3]
LDAPUR   X12, [X3, #-256]
LDAPUR   X13, [X4, #255]
ldapur   x14, [sp, #9]

//CHECK:       ldapur    x12, [x3]             // encoding: [0x6c,0x00,0x40,0xd9]
//CHECK-NEXT:  ldapur  x12, [x3, #-256]        // encoding: [0x6c,0x00,0x50,0xd9]
//CHECK-NEXT:  ldapur  x13, [x4, #255]         // encoding: [0x8d,0xf0,0x4f,0xd9]
//CHECK-NEXT:  ldapur  x14, [sp, #9]           // encoding: [0xee,0x93,0x40,0xd9]

//CHECK-NO-V84:      error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLURB   WZR, [X10]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLURB   W1, [X10]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLURB   W1, [X10, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: stlurb   w2, [x11, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLURB   W3, [SP, #-3]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapurb  wzr, [x12]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapurb  w4, [x12]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapurb  w4, [x12, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURB  W5, [X13, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURB  W6, [SP, #-2]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSB W7, [X14]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSB W7, [X14, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursb w8, [x15, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursb w9, [sp, #-1]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSB X0, [X16]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSB X0, [X16, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSB X1, [X17, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursb x2, [sp, #0]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursb x2, [sp]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: stlurh   w10, [x18]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: stlurh   w10, [x18, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLURH   W11, [X19, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLURH   W12, [SP, #1]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURH  W13, [X20]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURH  W13, [X20, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapurh  w14, [x21, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURH  W15, [SP, #2]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSH W16, [X22]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSH W16, [X22, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSH W17, [X23, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursh w18, [sp, #3]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursh x3, [x24]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursh x3, [x24, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSH X4, [X25, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSH X5, [SP, #4]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLUR    W19, [X26]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLUR    W19, [X26, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: stlur    w20, [x27, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLUR    W21, [SP, #5]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPUR   W22, [X28]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPUR   W22, [X28, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPUR   W23, [X29, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapur   w24, [sp, #6]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursw x6, [x30]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapursw x6, [x30, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSW X7, [X0, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPURSW X8, [SP, #7]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLUR    X9, [X1]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLUR    X9, [X1, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: stlur    x10, [x2, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: STLUR    X11, [SP, #8]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPUR   X12, [X3]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPUR   X12, [X3, #-256]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: LDAPUR   X13, [X4, #255]
//CHECK-NO-V84-NEXT: ^
//CHECK-NO-V84-NEXT: error: instruction requires: armv8.4a
//CHECK-NO-V84-NEXT: ldapur   x14, [sp, #9]
//CHECK-NO-V84-NEXT: ^
