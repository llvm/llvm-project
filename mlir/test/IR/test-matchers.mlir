// RUN: mlir-opt %s -disable-pass-threading=true -test-matchers -o /dev/null 2>&1 | FileCheck %s

func @test1(%a: f32, %b: f32, %c: f32) {
  %0 = addf %a, %b: f32
  %1 = addf %a, %c: f32
  %2 = addf %c, %b: f32
  %3 = mulf %a, %2: f32
  %4 = mulf %3, %1: f32
  %5 = mulf %4, %4: f32
  %6 = mulf %5, %5: f32
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
  %0 = constant 1.0: f32
  %1 = addf %a, %0: f32
  %2 = mulf %a, %1: f32
  return %2: f32
}

// CHECK-LABEL: test2
//       CHECK:   Pattern add(add(a, constant), a) matched and bound constant to: 1.000000e+00
