// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s | FileCheck %s --check-prefix=CHECK
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// Armv8.4-A flag manipulation instructions
//------------------------------------------------------------------------------

  cfinv
  setf8 w1
  setf8 wzr
  setf16 w1
  setf16 wzr
  rmif x1, #63, #15
  rmif xzr, #63, #15

//CHECK:      cfinv                        // encoding: [0x1f,0x40,0x00,0xd5]
//CHECK-NEXT: setf8 w1                     // encoding: [0x2d,0x08,0x00,0x3a]
//CHECK-NEXT: setf8 wzr                    // encoding: [0xed,0x0b,0x00,0x3a]
//CHECK-NEXT: setf16 w1                    // encoding: [0x2d,0x48,0x00,0x3a]
//CHECK-NEXT: setf16 wzr                   // encoding: [0xed,0x4b,0x00,0x3a]
//CHECK-NEXT: rmif x1, #63, #15            // encoding: [0x2f,0x84,0x1f,0xba]
//CHECK-NEXT: rmif xzr, #63, #15           // encoding: [0xef,0x87,0x1f,0xba]

//CHECK-ERROR:      error: instruction requires: armv8.4a
//CHECK-ERROR-NEXT: cfinv
//CHECK-ERROR-NEXT: ^
//CHECK-ERROR-NEXT: error: instruction requires: armv8.4a
//CHECK-ERROR-NEXT: setf8 w1
//CHECK-ERROR-NEXT: ^
//CHECK-ERROR-NEXT: error: instruction requires: armv8.4a
//CHECK-ERROR-NEXT: setf8 wzr
//CHECK-ERROR-NEXT: ^
//CHECK-ERROR-NEXT: error: instruction requires: armv8.4a
//CHECK-ERROR-NEXT: setf16 w1
//CHECK-ERROR-NEXT: ^
//CHECK-ERROR-NEXT: error: instruction requires: armv8.4a
//CHECK-ERROR-NEXT: setf16 wzr
//CHECK-ERROR-NEXT: ^
//CHECK-ERROR-NEXT: error: instruction requires: armv8.4a
//CHECK-ERROR-NEXT: rmif x1, #63, #15
//CHECK-ERROR-NEXT: ^
//CHECK-ERROR-NEXT: error: instruction requires: armv8.4a
//CHECK-ERROR-NEXT: rmif xzr, #63, #15
//CHECK-ERROR-NEXT: ^
