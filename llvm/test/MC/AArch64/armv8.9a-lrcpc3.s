// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding               -mattr=+rcpc3 < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.9a -mattr=+rcpc3 < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.4a -mattr=+rcpc3 < %s | FileCheck %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu               < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-RCPC3 %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.9a < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-RCPC3 %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v9.4a < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-RCPC3 %s

               stilp   w24, w0, [x16, #-8]!
// CHECK:      stilp   w24, w0, [x16, #-8]!     // encoding: [0x18,0x0a,0x00,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stilp   w24, w0, [x16,  -8]!
// CHECK:      stilp   w24, w0, [x16, #-8]!     // encoding: [0x18,0x0a,0x00,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stilp   x25, x1, [x17,  -16]!
// CHECK:      stilp   x25, x1, [x17, #-16]!    // encoding: [0x39,0x0a,0x01,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stilp   x25, x1, [x17, #-16]!
// CHECK:      stilp   x25, x1, [x17, #-16]!    // encoding: [0x39,0x0a,0x01,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stilp   w26, w2, [x18]
// CHECK:      stilp   w26, w2, [x18]           // encoding: [0x5a,0x1a,0x02,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stilp   w26, w2, [x18, #0]
// CHECK:      stilp   w26, w2, [x18]           // encoding: [0x5a,0x1a,0x02,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stilp   x27, x3, [sp]
// CHECK:      stilp   x27, x3, [sp]            // encoding: [0xfb,0x1b,0x03,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stilp   x27, x3, [sp, 0]
// CHECK:      stilp   x27, x3, [sp]            // encoding: [0xfb,0x1b,0x03,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  w28, w4, [x20], #8
// CHECK:      ldiapp  w28, w4, [x20], #8       // encoding: [0x9c,0x0a,0x44,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  w28, w4, [x20, #0], #8
// CHECK:      ldiapp  w28, w4, [x20], #8       // encoding: [0x9c,0x0a,0x44,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  w28, w4, [x20],  8
// CHECK:      ldiapp  w28, w4, [x20], #8       // encoding: [0x9c,0x0a,0x44,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  w28, w4, [x20, 0],  8
// CHECK:      ldiapp  w28, w4, [x20], #8       // encoding: [0x9c,0x0a,0x44,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  x29, x5, [x21], #16
// CHECK:      ldiapp  x29, x5, [x21], #16      // encoding: [0xbd,0x0a,0x45,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  x29, x5, [x21],  16
// CHECK:      ldiapp  x29, x5, [x21], #16      // encoding: [0xbd,0x0a,0x45,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  w30, w6, [sp]
// CHECK:      ldiapp  w30, w6, [sp]            // encoding: [0xfe,0x1b,0x46,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  w30, w6, [sp, #0]
// CHECK:      ldiapp  w30, w6, [sp]            // encoding: [0xfe,0x1b,0x46,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  xzr, x7, [x23]
// CHECK:      ldiapp  xzr, x7, [x23]           // encoding: [0xff,0x1a,0x47,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldiapp  xzr, x7, [x23, 0]
// CHECK:      ldiapp  xzr, x7, [x23]           // encoding: [0xff,0x1a,0x47,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3

               stlr w3, [x15, #-4]!
// CHECK:      stlr w3, [x15, #-4]!    // encoding: [0xe3,0x09,0x80,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stlr w3, [x15,  -4]!
// CHECK:      stlr w3, [x15, #-4]!    // encoding: [0xe3,0x09,0x80,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stlr x3, [x15, #-8]!
// CHECK:      stlr x3, [x15, #-8]!    // encoding: [0xe3,0x09,0x80,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stlr x3, [sp,  -8]!
// CHECK:      stlr x3, [sp, #-8]!     // encoding: [0xe3,0x0b,0x80,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapr w3, [sp], #4
// CHECK:      ldapr w3, [sp], #4       // encoding: [0xe3,0x0b,0xc0,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapr w3, [x15], 4
// CHECK:      ldapr w3, [x15], #4      // encoding: [0xe3,0x09,0xc0,0x99]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapr x3, [x15], #8
// CHECK:      ldapr x3, [x15], #8      // encoding: [0xe3,0x09,0xc0,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapr x3, [x15], 8
// CHECK:      ldapr x3, [x15], #8      // encoding: [0xe3,0x09,0xc0,0xd9]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3

               stlur b3, [x15, #-1]
// CHECK:      stlur b3, [x15, #-1]  // encoding: [0xe3,0xf9,0x1f,0x1d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stlur h3, [x15, #2]
// CHECK:      stlur h3, [x15, #2]   // encoding: [0xe3,0x29,0x00,0x5d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stlur s3, [x15, #-3]
// CHECK:      stlur s3, [x15, #-3]  // encoding: [0xe3,0xd9,0x1f,0x9d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stlur d3, [sp, #4]
// CHECK:      stlur d3, [sp, #4]    // encoding: [0xe3,0x4b,0x00,0xdd]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stlur q3, [x15, #-5]
// CHECK:      stlur q3, [x15, #-5]  // encoding: [0xe3,0xb9,0x9f,0x1d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapur b3, [x15, #6]
// CHECK:      ldapur b3, [x15, #6]  // encoding: [0xe3,0x69,0x40,0x1d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapur h3, [x15, #-7]
// CHECK:      ldapur h3, [x15, #-7] // encoding: [0xe3,0x99,0x5f,0x5d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapur s3, [x15, #8]
// CHECK:      ldapur s3, [x15, #8]  // encoding: [0xe3,0x89,0x40,0x9d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapur d3, [x15, #-9]
// CHECK:      ldapur d3, [x15, #-9] // encoding: [0xe3,0x79,0x5f,0xdd]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldapur q3, [sp, #10]
// CHECK:      ldapur q3, [sp, #10]  // encoding: [0xe3,0xab,0xc0,0x1d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3

               stl1  { v3.d }[0], [x15]
// CHECK:      stl1  { v3.d }[0], [x15]     // encoding: [0xe3,0x85,0x01,0x0d]
// ERROR-NO-RCPC3:  [[@LINE-2]]:16: error: instruction requires: rcpc3
               stl1  { v3.d }[0], [x15, #0]
// CHECK:      stl1  { v3.d }[0], [x15]     // encoding: [0xe3,0x85,0x01,0x0d]
// ERROR-NO-RCPC3:  [[@LINE-2]]:16: error: instruction requires: rcpc3
               stl1  { v3.d }[1], [sp]
// CHECK:      stl1  { v3.d }[1], [sp]      // encoding: [0xe3,0x87,0x01,0x4d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               stl1  { v3.d }[1], [sp, 0]
// CHECK:      stl1  { v3.d }[1], [sp]      // encoding: [0xe3,0x87,0x01,0x4d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldap1 { v3.d }[0], [sp]
// CHECK:      ldap1 { v3.d }[0], [sp]      // encoding: [0xe3,0x87,0x41,0x0d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldap1 { v3.d }[0], [sp, #0]
// CHECK:      ldap1 { v3.d }[0], [sp]      // encoding: [0xe3,0x87,0x41,0x0d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldap1 { v3.d }[1], [x15]
// CHECK:      ldap1 { v3.d }[1], [x15]     // encoding: [0xe3,0x85,0x41,0x4d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
               ldap1 { v3.d }[1], [x15, 0]
// CHECK:      ldap1 { v3.d }[1], [x15]     // encoding: [0xe3,0x85,0x41,0x4d]
// ERROR-NO-RCPC3: [[@LINE-2]]:16: error: instruction requires: rcpc3
