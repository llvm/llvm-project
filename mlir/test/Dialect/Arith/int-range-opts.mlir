// RUN: mlir-opt -int-range-optimizations --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test() -> i1 {
  %cst1 = arith.constant -1 : index
  %0 = test.with_bounds { umin = 0 : index, umax = 0x7fffffffffffffff : index, smin = 0 : index, smax = 0x7fffffffffffffff : index } : index
  %1 = arith.cmpi eq, %0, %cst1 : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test() -> i1 {
  %cst1 = arith.constant -1 : index
  %0 = test.with_bounds { umin = 0 : index, umax = 0x7fffffffffffffff : index, smin = 0 : index, smax = 0x7fffffffffffffff : index } : index
  %1 = arith.cmpi ne, %0, %cst1 : index
  return %1: i1
}

// -----


// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test() -> i1 {
  %cst = arith.constant 0 : index
  %0 = test.with_bounds { umin = 0 : index, umax = 0x7fffffffffffffff : index, smin = 0 : index, smax = 0x7fffffffffffffff : index } : index
  %1 = arith.cmpi sge, %0, %cst : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test() -> i1 {
  %cst = arith.constant 0 : index
  %0 = test.with_bounds { umin = 0 : index, umax = 0x7fffffffffffffff : index, smin = 0 : index, smax = 0x7fffffffffffffff : index } : index
  %1 = arith.cmpi slt, %0, %cst : index
  return %1: i1
}

// -----


// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test() -> i1 {
  %cst1 = arith.constant -1 : index
  %0 = test.with_bounds { umin = 0 : index, umax = 0x7fffffffffffffff : index, smin = 0 : index, smax = 0x7fffffffffffffff : index } : index
  %1 = arith.cmpi sgt, %0, %cst1 : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test() -> i1 {
  %cst1 = arith.constant -1 : index
  %0 = test.with_bounds { umin = 0 : index, umax = 0x7fffffffffffffff : index, smin = 0 : index, smax = 0x7fffffffffffffff : index } : index
  %1 = arith.cmpi sle, %0, %cst1 : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
// CHECK: test.reflect_bounds {smax = 24 : si8, smin = 0 : si8, umax = 24 : ui8, umin = 0 : ui8}
func.func @test() -> i8 {
  %cst1 = arith.constant 1 : i8
  %i8val = test.with_bounds { umin = 0 : i8, umax = 12 : i8, smin = 0 : i8, smax = 12 : i8 } : i8
  %shifted = arith.shli %i8val, %cst1 : i8
  %1 = test.reflect_bounds %shifted : i8
  return %1: i8
}

// -----

// CHECK-LABEL: func @test
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = -128 : si8, umax = 254 : ui8, umin = 0 : ui8}
func.func @test() -> i8 {
  %cst1 = arith.constant 1 : i8
  %i8val = test.with_bounds { umin = 0 : i8, umax = 127 : i8, smin = 0 : i8, smax = 127 : i8 } : i8
  %shifted = arith.shli %i8val, %cst1 : i8
  %1 = test.reflect_bounds %shifted : i8
  return %1: i8
}

