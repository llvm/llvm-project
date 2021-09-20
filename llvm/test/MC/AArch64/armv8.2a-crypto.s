// RUN: not llvm-mc -triple aarch64 -mattr=+sha3,-sm4 -show-encoding < %s 2> %t | FileCheck %s  --check-prefix=CHECK-SHA
// RUN: FileCheck --check-prefix=CHECK-NO-SM < %t %s

// RUN: not llvm-mc -triple aarch64 -mattr=+sm4,-sha3 -show-encoding < %s 2> %t | FileCheck %s --check-prefix=CHECK-SM
// RUN: FileCheck --check-prefix=CHECK-NO-SHA < %t %s

// RUN: not llvm-mc -triple aarch64 -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-SHA --check-prefix=CHECK-NO-SM < %t %s

// RUN: llvm-mc -triple aarch64 -mattr=+v8r -show-encoding -o - %s | FileCheck %s --check-prefixes=CHECK-SM,CHECK-SHA

  sha512h   q0, q1, v2.2d
  sha512h2  q0, q1, v2.2d
  sha512su0 v11.2d, v12.2d
  sha512su1 v11.2d, v13.2d, v14.2d
  eor3  v25.16b, v12.16b, v7.16b, v2.16b
  rax1  v30.2d, v29.2d, v26.2d
  xar v26.2d, v21.2d, v27.2d, #63
  bcax  v31.16b, v26.16b, v2.16b, v1.16b

//CHECK-SHA:  sha512h   q0, q1, v2.2d                    // encoding: [0x20,0x80,0x62,0xce]
//CHECK-SHA:  sha512h2  q0, q1, v2.2d                    // encoding: [0x20,0x84,0x62,0xce]
//CHECK-SHA:  sha512su0 v11.2d, v12.2d                   // encoding: [0x8b,0x81,0xc0,0xce]
//CHECK-SHA:  sha512su1 v11.2d, v13.2d, v14.2d           // encoding: [0xab,0x89,0x6e,0xce]
//CHECK-SHA:  eor3  v25.16b, v12.16b, v7.16b, v2.16b     // encoding: [0x99,0x09,0x07,0xce]
//CHECK-SHA:  rax1  v30.2d, v29.2d, v26.2d               // encoding: [0xbe,0x8f,0x7a,0xce]
//CHECK-SHA:  xar v26.2d, v21.2d, v27.2d, #63            // encoding: [0xba,0xfe,0x9b,0xce]
//CHECK-SHA:  bcax  v31.16b, v26.16b, v2.16b, v1.16b     // encoding: [0x5f,0x07,0x22,0xce]


// CHECK-NO-SHA: error: instruction requires: sha3
// CHECK-NO-SHA: error: instruction requires: sha3
// CHECK-NO-SHA: error: instruction requires: sha3
// CHECK-NO-SHA: error: instruction requires: sha3
// CHECK-NO-SHA: error: instruction requires: sha3
// CHECK-NO-SHA: error: instruction requires: sha3
// CHECK-NO-SHA: error: instruction requires: sha3
// CHECK-NO-SHA: error: instruction requires: sha3

  sm3ss1  v20.4s, v23.4s, v21.4s, v22.4s
  sm3tt1a v20.4s, v23.4s, v21.s[3]
  sm3tt1b v20.4s, v23.4s, v21.s[3]
  sm3tt2a v20.4s, v23.4s, v21.s[3]
  sm3tt2b v20.4s, v23.4s, v21.s[3]
  sm3partw1 v30.4s, v29.4s, v26.4s
  sm3partw2 v30.4s, v29.4s, v26.4s
  sm4ekey v11.4s, v11.4s, v19.4s
  sm4e  v2.4s, v15.4s

// CHECK-SM:  sm3ss1  v20.4s, v23.4s, v21.4s, v22.4s     // encoding: [0xf4,0x5a,0x55,0xce]
// CHECK-SM:  sm3tt1a v20.4s, v23.4s, v21.s[3]           // encoding: [0xf4,0xb2,0x55,0xce]
// CHECK-SM:  sm3tt1b v20.4s, v23.4s, v21.s[3]           // encoding: [0xf4,0xb6,0x55,0xce]
// CHECK-SM:  sm3tt2a v20.4s, v23.4s, v21.s[3]           // encoding: [0xf4,0xba,0x55,0xce]
// CHECK-SM:  sm3tt2b v20.4s, v23.4s, v21.s[3]           // encoding: [0xf4,0xbe,0x55,0xce]
// CHECK-SM:  sm3partw1 v30.4s, v29.4s, v26.4s           // encoding: [0xbe,0xc3,0x7a,0xce]
// CHECK-SM:  sm3partw2 v30.4s, v29.4s, v26.4s           // encoding: [0xbe,0xc7,0x7a,0xce]
// CHECK-SM:  sm4ekey v11.4s, v11.4s, v19.4s             // encoding: [0x6b,0xc9,0x73,0xce]
// CHECK-SM:  sm4e v2.4s, v15.4s                         // encoding: [0xe2,0x85,0xc0,0xce]

// CHECK-NO-SM: error: instruction requires: sm4
// CHECK-NO-SM: error: instruction requires: sm4
// CHECK-NO-SM: error: instruction requires: sm4
// CHECK-NO-SM: error: instruction requires: sm4
// CHECK-NO-SM: error: instruction requires: sm4
// CHECK-NO-SM: error: instruction requires: sm4
// CHECK-NO-SM: error: instruction requires: sm4
