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

// -----

// CHECK-LABEL: func @trivial_rem
// CHECK: [[val:%.+]] = test.with_bounds
// CHECK: return [[val]]
func.func @trivial_rem() -> i8 {
  %c64 = arith.constant 64 : i8
  %val = test.with_bounds { umin = 0 : ui8, umax = 63 : ui8, smin = 0 : si8, smax = 63 : si8 } : i8
  %mod = arith.remsi %val, %c64 : i8
  return %mod : i8
}

// -----

// CHECK-LABEL: func @non_const_rhs
// CHECK: [[mod:%.+]] = arith.remui
// CHECK: return [[mod]]
func.func @non_const_rhs() -> i8 {
  %c64 = arith.constant 64 : i8
  %val = test.with_bounds { umin = 0 : ui8, umax = 2 : ui8, smin = 0 : si8, smax = 2 : si8 } : i8
  %rhs = test.with_bounds { umin = 63 : ui8, umax = 64 : ui8, smin = 63 : si8, smax = 64 : si8 } : i8
  %mod = arith.remui %val, %rhs : i8
  return %mod : i8
}

// -----

// CHECK-LABEL: func @wraps
// CHECK: [[mod:%.+]] = arith.remsi
// CHECK: return [[mod]]
func.func @wraps() -> i8 {
  %c64 = arith.constant 64 : i8
  %val = test.with_bounds { umin = 63 : ui8, umax = 65 : ui8, smin = 63 : si8, smax = 65 : si8 } : i8
  %mod = arith.remsi %val, %c64 : i8
  return %mod : i8
}

// -----

// CHECK-LABEL: @analysis_crash
func.func @analysis_crash(%arg0: i32, %arg1: tensor<128xi1>) -> tensor<128xi64> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<-1> : tensor<128xi32>
  %splat = tensor.splat %arg0 : tensor<128xi32>
  %0 = scf.for %arg2 = %c0_i32 to %arg0 step %arg0 iter_args(%arg3 = %splat) -> (tensor<128xi32>)  : i32 {
    scf.yield %arg3 : tensor<128xi32>
  }
  %1 = arith.select %arg1, %0#0, %cst : tensor<128xi1>, tensor<128xi32>
  // Make sure the analysis doesn't crash when materializing the range as a tensor constant.
  %2 = arith.extsi %1 : tensor<128xi32> to tensor<128xi64>
  return %2 : tensor<128xi64>
}
