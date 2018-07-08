// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK

//------------------------------------------------------------------------------
// Armv8.4-A flag manipulation instructions
//------------------------------------------------------------------------------

  rmif  x1, #64, #15
  rmif  x1, #-1, #15
  rmif  x1, #63, #16
  rmif  x1, #63, #-1
  rmif  sp, #63, #1

//CHECK:      error: immediate must be an integer in range [0, 63].
//CHECK-NEXT: rmif  x1, #64, #15
//CHECK-NEXT:           ^
//CHECK-NEXT: error: immediate must be an integer in range [0, 63].
//CHECK-NEXT: rmif  x1, #-1, #15
//CHECK-NEXT:           ^
//CHECK-NEXT: error: immediate must be an integer in range [0, 15].
//CHECK-NEXT: rmif  x1, #63, #16
//CHECK-NEXT:                ^
//CHECK-NEXT: error: immediate must be an integer in range [0, 15].
//CHECK-NEXT: rmif  x1, #63, #-1
//CHECK-NEXT:                ^
//CHECK-NEXT: error: invalid operand for instruction
//CHECK-NEXT: rmif  sp, #63, #1
//CHECK-NEXT:       ^
