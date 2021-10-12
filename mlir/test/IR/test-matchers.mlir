// RUN: mlir-opt %s -mlir-disable-threading=true -test-matchers -o /dev/null 2>&1 | FileCheck %s

func @test1(%a: f32, %b: f32, %c: f32) {
  %0 = arith.addf %a, %b: f32
  %1 = arith.addf %a, %c: f32
  %2 = arith.addf %c, %b: f32
  %3 = arith.mulf %a, %2: f32
  %4 = arith.mulf %3, %1: f32
  %5 = arith.mulf %4, %4: f32
  %6 = arith.mulf %5, %5: f32
  return
}

// CHECK-LABEL: test1
//       CHECK:   Pattern add(*) matched 3 times
//       CHECK:   Pattern mul(*) matched 4 times
//       CHECK:   Pattern add(add(*), *) matched 0 times
//       CHECK:   Pattern add(*, add(*)) matched 0 times
//       CHECK:   Pattern mul(add(*), *) matched 0 times
//       CHECK:   Pattern mul(*, add(*)) matched 2 times
//       CHECK:   Pattern mul(mul(*), *) matched 3 times
//       CHECK:   Pattern mul(mul(*), mul(*)) matched 2 times
//       CHECK:   Pattern mul(mul(mul(*), mul(*)), mul(mul(*), mul(*))) matched 1 times
//       CHECK:   Pattern mul(mul(mul(mul(*), add(*)), mul(*)), mul(mul(*, add(*)), mul(*, add(*)))) matched 1 times
//       CHECK:   Pattern add(a, b) matched 1 times
//       CHECK:   Pattern add(a, c) matched 1 times
//       CHECK:   Pattern add(b, a) matched 0 times
//       CHECK:   Pattern add(c, a) matched 0 times
//       CHECK:   Pattern mul(a, add(c, b)) matched 1 times
//       CHECK:   Pattern mul(a, add(b, c)) matched 0 times
//       CHECK:   Pattern mul(mul(a, *), add(a, c)) matched 1 times
//       CHECK:   Pattern mul(mul(a, *), add(c, b)) matched 0 times

func @test2(%a: f32) -> f32 {
  %0 = arith.constant 1.0: f32
  %1 = arith.addf %a, %0: f32
  %2 = arith.mulf %a, %1: f32
  return %2: f32
}

// CHECK-LABEL: test2
//       CHECK:   Pattern add(add(a, constant), a) matched and bound constant to: 1.000000e+00
//       CHECK:   Pattern add(add(a, constant), a) matched
