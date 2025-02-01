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

// Note: I wish I had a simpler example than this, but getting rid of a
// bunch of the arithmetic made the issue go away.
// CHECK-LABEL: @blocks_prematurely_declared_dead_bug
// CHECK-NOT: arith.constant true
func.func @blocks_prematurely_declared_dead_bug(%mem: memref<?xf16>) {
  %cst = arith.constant dense<false> : vector<1xi1>
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
  %cst_1 = arith.constant 0.000000e+00 : f16
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %thread_id_x = gpu.thread_id  x upper_bound 64
  %6 = test.with_bounds { smin = 16 : index, smax = 112 : index, umin = 16 : index, umax = 112 : index } : index
  %8 = arith.divui %6, %c16 : index
  %9 = arith.muli %8, %c16 : index
  cf.br ^bb1(%c0 : index)
^bb1(%12: index):  // 2 preds: ^bb0, ^bb7
  %13 = arith.cmpi slt, %12, %9 : index
  cf.cond_br %13, ^bb2, ^bb8
^bb2:  // pred: ^bb1
  %14 = arith.subi %9, %12 : index
  %15 = arith.minsi %14, %c64 : index
  %16 = arith.subi %15, %thread_id_x : index
  %17 = vector.constant_mask [1] : vector<1xi1>
  %18 = arith.cmpi sgt, %16, %c0 : index
  %19 = arith.select %18, %17, %cst : vector<1xi1>
  %20 = vector.extract %19[0] : i1 from vector<1xi1>
  %21 = vector.insert %20, %cst [0] : i1 into vector<1xi1>
  %22 = arith.addi %12, %thread_id_x : index
  cf.br ^bb3(%c0, %cst_0 : index, vector<1xf16>)
^bb3(%23: index, %24: vector<1xf16>):  // 2 preds: ^bb2, ^bb6
  %25 = arith.cmpi slt, %23, %c1 : index
  cf.cond_br %25, ^bb4, ^bb7
^bb4:  // pred: ^bb3
  %26 = vector.extractelement %21[%23 : index] : vector<1xi1>
  cf.cond_br %26, ^bb5, ^bb6(%24 : vector<1xf16>)
^bb5:  // pred: ^bb4
  %27 = arith.addi %22, %23 : index
  %28 = memref.load %mem[%27] : memref<?xf16>
  %29 = vector.insertelement %28, %24[%23 : index] : vector<1xf16>
  cf.br ^bb6(%29 : vector<1xf16>)
^bb6(%30: vector<1xf16>):  // 2 preds: ^bb4, ^bb5
  %31 = arith.addi %23, %c1 : index
  cf.br ^bb3(%31, %30 : index, vector<1xf16>)
^bb7:  // pred: ^bb3
  %37 = arith.addi %12, %c64 : index
  cf.br ^bb1(%37 : index)
^bb8:  // pred: ^bb1
  %70 = arith.cmpi eq, %thread_id_x, %c0 : index
  cf.cond_br %70, ^bb9, ^bb10
^bb9:  // pred: ^bb8
  memref.store %cst_1, %mem[%c0] : memref<?xf16>
  cf.br ^bb10
^bb10:  // 2 preds: ^bb8, ^bb9
  return
}
