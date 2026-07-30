// RUN: llvm-mc -output-asm-variant=1 -triple aarch64-apple-ios -mattr=+sha3,+sm4 -show-encoding < %s | FileCheck %s

  sha512h.2d   q0, q1, v2
  sha512h2.2d  q0, q1, v2
  sha512su0.2d v11, v12
  sha512su1.2d v11, v13, v14
  eor3.16b  v25, v12, v7, v2
  rax1.2d  v30, v29, v26
  xar.2d v26, v21, v27, #63
  bcax.16b  v31, v26, v2, v1

//CHECK:  sha512h.2d   q0, q1, v2                    ; encoding: [0x20,0x80,0x62,0xce]
//CHECK:  sha512h2.2d  q0, q1, v2                    ; encoding: [0x20,0x84,0x62,0xce]
//CHECK:  sha512su0.2d v11, v12                      ; encoding: [0x8b,0x81,0xc0,0xce]
//CHECK:  sha512su1.2d v11, v13, v14                 ; encoding: [0xab,0x89,0x6e,0xce]
//CHECK:  eor3.16b  v25, v12, v7, v2                 ; encoding: [0x99,0x09,0x07,0xce]
//CHECK:  rax1.2d  v30, v29, v26                     ; encoding: [0xbe,0x8f,0x7a,0xce]
//CHECK:  xar.2d v26, v21, v27, #63                  ; encoding: [0xba,0xfe,0x9b,0xce]
//CHECK:  bcax.16b  v31, v26, v2, v1                 ; encoding: [0x5f,0x07,0x22,0xce]



  sm3ss1.4s  v20, v23, v21, v22
  sm3tt1a.4s v20, v23, v21[3]
  sm3tt1b.4s v20, v23, v21[3]
  sm3tt2a.4s v20, v23, v21[3]
  sm3tt2b.4s v20, v23, v21[3]
  sm3partw1.4s v30, v29, v26
  sm3partw2.4s v30, v29, v26
  sm4ekey.4s v11, v11, v19
  sm4e.4s  v2, v15

// CHECK:  sm3ss1.4s  v20, v23, v21, v22             ; encoding: [0xf4,0x5a,0x55,0xce]
// CHECK:  sm3tt1a.4s v20, v23, v21[3]               ; encoding: [0xf4,0xb2,0x55,0xce]
// CHECK:  sm3tt1b.4s v20, v23, v21[3]               ; encoding: [0xf4,0xb6,0x55,0xce]
// CHECK:  sm3tt2a.4s v20, v23, v21[3]               ; encoding: [0xf4,0xba,0x55,0xce]
// CHECK:  sm3tt2b.4s v20, v23, v21[3]               ; encoding: [0xf4,0xbe,0x55,0xce]
// CHECK:  sm3partw1.4s v30, v29, v26                ; encoding: [0xbe,0xc3,0x7a,0xce]
// CHECK:  sm3partw2.4s v30, v29, v26                ; encoding: [0xbe,0xc7,0x7a,0xce]
// CHECK:  sm4ekey.4s v11, v11, v19                  ; encoding: [0x6b,0xc9,0x73,0xce]
// CHECK:  sm4e.4s v2, v15                           ; encoding: [0xe2,0x85,0xc0,0xce]
