// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}))' | FileCheck %s

// Test case: Simple case of deleting a dead pure op.

// CHECK:      func @f(%arg0: f32) {
// CHECK-NEXT:   return

func.func @f(%arg0: f32) {
  %0 = "arith.addf"(%arg0, %arg0) : (f32, f32) -> f32
  return
}

// -----

// Test case: Simple case of deleting a block argument.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   "test.br"()[^bb1]
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   return

func.func @f(%arg0: f32) {
  "test.br"(%arg0)[^succ] : (f32) -> ()
^succ(%0: f32):
  return
}

// -----

// Test case: Deleting recursively dead block arguments.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.br ^bb1


func.func @f(%arg0: f32) {
  cf.br ^loop(%arg0: f32)
^loop(%loop: f32):
  cf.br ^loop(%loop: f32)
}

// -----

// Test case: Deleting recursively dead block arguments with pure ops in between.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.br ^bb1

func.func @f(%arg0: f32) {
  cf.br ^loop(%arg0: f32)
^loop(%0: f32):
  %1 = "math.exp"(%0) : (f32) -> f32
  cf.br ^loop(%1: f32)
}

// -----

// Test case: Delete block arguments for cf.cond_br.

// CHECK:      func @f(%arg0: f32, %arg1: i1)
// CHECK-NEXT:   return

func.func @f(%arg0: f32, %pred: i1) {
  %exp = "math.exp"(%arg0) : (f32) -> f32
  cf.cond_br %pred, ^true(%exp: f32), ^false(%exp: f32)
^true(%0: f32):
  return
^false(%1: f32):
  return
}

// -----

// Test case: Recursively DCE into enclosed regions.

// CHECK:      func.func @f(%arg0: f32)
// CHECK-NOT:     arith.addf

func.func @f(%arg0: f32) {
  "test.region"() (
    {
      %0 = "arith.addf"(%arg0, %arg0) : (f32, f32) -> f32
    }
  ) : () -> ()
  return
}

// -----

// Test case: Don't delete pure ops that feed into returns.

// CHECK:      func @f(%arg0: f32) -> f32
// CHECK-NEXT:   [[VAL0:%.+]] = arith.addf %arg0, %arg0 : f32
// CHECK-NEXT:   return [[VAL0]] : f32

func.func @f(%arg0: f32) -> f32 {
  %0 = "arith.addf"(%arg0, %arg0) : (f32, f32) -> f32
  return %0 : f32
}

// -----

// Test case: Don't delete potentially side-effecting ops.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   "foo.print"(%arg0) : (f32) -> ()
// CHECK-NEXT:   return

func.func @f(%arg0: f32) {
  "foo.print"(%arg0) : (f32) -> ()
  return
}

// -----

// Test case: Uses in nested regions are deleted correctly.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   "foo.has_region"
// CHECK-NEXT:     "foo.return"

func.func @f(%arg0: f32) {
  %0 = "math.exp"(%arg0) : (f32) -> f32
  "foo.has_region"() ({
    %1 = "math.exp"(%0) : (f32) -> f32
    "foo.return"() : () -> ()
  }) : () -> ()
  return
}

// -----

// Test case: Test the mechanics of deleting multiple block arguments.

// CHECK:      func @f(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>, %arg2: tensor<3xf32>, %arg3: tensor<4xf32>, %arg4: tensor<5xf32>)
// CHECK-NEXT:   "test.br"(%arg1, %arg3)[^bb1] : (tensor<2xf32>, tensor<4xf32>)
// CHECK-NEXT: ^bb1([[VAL0:%.+]]: tensor<2xf32>, [[VAL1:%.+]]: tensor<4xf32>):
// CHECK-NEXT:   "foo.print"([[VAL0]])
// CHECK-NEXT:   "foo.print"([[VAL1]])
// CHECK-NEXT:   return


func.func @f(
  %arg0: tensor<1xf32>,
  %arg1: tensor<2xf32>,
  %arg2: tensor<3xf32>,
  %arg3: tensor<4xf32>,
  %arg4: tensor<5xf32>) {
  "test.br"(%arg0, %arg1, %arg2, %arg3, %arg4)[^succ] : (tensor<1xf32>, tensor<2xf32>, tensor<3xf32>, tensor<4xf32>, tensor<5xf32>) -> ()
^succ(%t1: tensor<1xf32>, %t2: tensor<2xf32>, %t3: tensor<3xf32>, %t4: tensor<4xf32>, %t5: tensor<5xf32>):
  "foo.print"(%t2) : (tensor<2xf32>) -> ()
  "foo.print"(%t4) : (tensor<4xf32>) -> ()
  return
}

// -----

// Test case: Test values with use-def cycles are deleted properly.

// CHECK:      func @f()
// CHECK-NEXT:   test.graph_region
// CHECK-NEXT:     "test.terminator"() : () -> ()

func.func @f() {
  test.graph_region {
    %0 = "math.exp"(%1) : (f32) -> f32
    %1 = "math.exp"(%0) : (f32) -> f32
    "test.terminator"() : ()->()
  }
  return
}

// -----


// Test case: Delete ops that only have side-effects on an allocated result.

// CHECK:      func @f()
// CHECK-NOT:    test_effects_result
// CHECK-NEXT:   return

func.func @f() {
  %0 = "test.test_effects_result"() : () -> i32
  return
}
