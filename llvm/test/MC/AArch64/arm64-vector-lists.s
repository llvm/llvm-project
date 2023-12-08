// RUN: not llvm-mc -triple arm64 -mattr=neon -show-encoding < %s 2>%t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

    ST4     {v0.8B-v3.8B}, [x0]
    ST4     {v0.4H-v3.4H}, [x0]

// CHECK: st4  { v0.8b, v1.8b, v2.8b, v3.8b }, [x0] // encoding: [0x00,0x00,0x00,0x0c]
// CHECK: st4  { v0.4h, v1.4h, v2.4h, v3.4h }, [x0] // encoding: [0x00,0x04,0x00,0x0c]

    ST4     {v0.8B-v4.8B}, [x0]
    ST4     {v0.8B-v3.8B,v4.8B}, [x0]
    ST4     {v0.8B-v3.8H}, [x0]
    ST4     {v0.8B-v3.16B}, [x0]
    ST4     {v0.8B-},[x0]

// CHECK-ERRORS: error: invalid number of vectors
// CHECK-ERRORS: error: '}' expected
// CHECK-ERRORS: error: mismatched register size suffix
// CHECK-ERRORS: error: mismatched register size suffix
// CHECK-ERRORS: error: vector register expected
