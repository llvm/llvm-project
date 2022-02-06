// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.func(cse)' | FileCheck %s

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0) -> (d0 mod 2)>
#map0 = affine_map<(d0) -> (d0 mod 2)>

// CHECK-LABEL: @simple_constant
func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
  %0 = arith.constant 1 : i32

  // CHECK-NEXT: return %c1_i32, %c1_i32 : i32, i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: @basic
func @basic() -> (index, index) {
  // CHECK: %c0 = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index

  // CHECK-NEXT: %0 = affine.apply #[[$MAP]](%c0)
  %0 = affine.apply #map0(%c0)
  %1 = affine.apply #map0(%c1)

  // CHECK-NEXT: return %0, %0 : index, index
  return %0, %1 : index, index
}

// CHECK-LABEL: @many
func @many(f32, f32) -> (f32) {
^bb0(%a : f32, %b : f32):
  // CHECK-NEXT: %0 = arith.addf %arg0, %arg1 : f32
  %c = arith.addf %a, %b : f32
  %d = arith.addf %a, %b : f32
  %e = arith.addf %a, %b : f32
  %f = arith.addf %a, %b : f32

  // CHECK-NEXT: %1 = arith.addf %0, %0 : f32
  %g = arith.addf %c, %d : f32
  %h = arith.addf %e, %f : f32
  %i = arith.addf %c, %e : f32

  // CHECK-NEXT: %2 = arith.addf %1, %1 : f32
  %j = arith.addf %g, %h : f32
  %k = arith.addf %h, %i : f32

  // CHECK-NEXT: %3 = arith.addf %2, %2 : f32
  %l = arith.addf %j, %k : f32

  // CHECK-NEXT: return %3 : f32
  return %l : f32
}

/// Check that operations are not eliminated if they have different operands.
// CHECK-LABEL: @different_ops
func @different_ops() -> (i32, i32) {
  // CHECK: %c0_i32 = arith.constant 0 : i32
  // CHECK: %c1_i32 = arith.constant 1 : i32
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32

  // CHECK-NEXT: return %c0_i32, %c1_i32 : i32, i32
  return %0, %1 : i32, i32
}

/// Check that operations are not eliminated if they have different result
/// types.
// CHECK-LABEL: @different_results
func @different_results(%arg0: tensor<*xf32>) -> (tensor<?x?xf32>, tensor<4x?xf32>) {
  // CHECK: %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  // CHECK-NEXT: %1 = tensor.cast %arg0 : tensor<*xf32> to tensor<4x?xf32>
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  %1 = tensor.cast %arg0 : tensor<*xf32> to tensor<4x?xf32>

  // CHECK-NEXT: return %0, %1 : tensor<?x?xf32>, tensor<4x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<4x?xf32>
}

/// Check that operations are not eliminated if they have different attributes.
// CHECK-LABEL: @different_attributes
func @different_attributes(index, index) -> (i1, i1, i1) {
^bb0(%a : index, %b : index):
  // CHECK: %0 = arith.cmpi slt, %arg0, %arg1 : index
  %0 = arith.cmpi slt, %a, %b : index

  // CHECK-NEXT: %1 = arith.cmpi ne, %arg0, %arg1 : index
  /// Predicate 1 means inequality comparison.
  %1 = arith.cmpi ne, %a, %b : index
  %2 = "arith.cmpi"(%a, %b) {predicate = 1} : (index, index) -> i1

  // CHECK-NEXT: return %0, %1, %1 : i1, i1, i1
  return %0, %1, %2 : i1, i1, i1
}

/// Check that operations with side effects are not eliminated.
// CHECK-LABEL: @side_effect
func @side_effect() -> (memref<2x1xf32>, memref<2x1xf32>) {
  // CHECK: %0 = memref.alloc() : memref<2x1xf32>
  %0 = memref.alloc() : memref<2x1xf32>

  // CHECK-NEXT: %1 = memref.alloc() : memref<2x1xf32>
  %1 = memref.alloc() : memref<2x1xf32>

  // CHECK-NEXT: return %0, %1 : memref<2x1xf32>, memref<2x1xf32>
  return %0, %1 : memref<2x1xf32>, memref<2x1xf32>
}

/// Check that operation definitions are properly propagated down the dominance
/// tree.
// CHECK-LABEL: @down_propagate_for
func @down_propagate_for() {
  // CHECK: %c1_i32 = arith.constant 1 : i32
  %0 = arith.constant 1 : i32

  // CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
  affine.for %i = 0 to 4 {
    // CHECK-NEXT: "foo"(%c1_i32, %c1_i32) : (i32, i32) -> ()
    %1 = arith.constant 1 : i32
    "foo"(%0, %1) : (i32, i32) -> ()
  }
  return
}

// CHECK-LABEL: @down_propagate
func @down_propagate() -> i32 {
  // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
  %0 = arith.constant 1 : i32

  // CHECK-NEXT: %true = arith.constant true
  %cond = arith.constant true

  // CHECK-NEXT: cf.cond_br %true, ^bb1, ^bb2(%c1_i32 : i32)
  cf.cond_br %cond, ^bb1, ^bb2(%0 : i32)

^bb1: // CHECK: ^bb1:
  // CHECK-NEXT: cf.br ^bb2(%c1_i32 : i32)
  %1 = arith.constant 1 : i32
  cf.br ^bb2(%1 : i32)

^bb2(%arg : i32):
  return %arg : i32
}

/// Check that operation definitions are NOT propagated up the dominance tree.
// CHECK-LABEL: @up_propagate_for
func @up_propagate_for() -> i32 {
  // CHECK: affine.for {{.*}} = 0 to 4 {
  affine.for %i = 0 to 4 {
    // CHECK-NEXT: %c1_i32_0 = arith.constant 1 : i32
    // CHECK-NEXT: "foo"(%c1_i32_0) : (i32) -> ()
    %0 = arith.constant 1 : i32
    "foo"(%0) : (i32) -> ()
  }

  // CHECK: %c1_i32 = arith.constant 1 : i32
  // CHECK-NEXT: return %c1_i32 : i32
  %1 = arith.constant 1 : i32
  return %1 : i32
}

// CHECK-LABEL: func @up_propagate
func @up_propagate() -> i32 {
  // CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
  %0 = arith.constant 0 : i32

  // CHECK-NEXT: %true = arith.constant true
  %cond = arith.constant true

  // CHECK-NEXT: cf.cond_br %true, ^bb1, ^bb2(%c0_i32 : i32)
  cf.cond_br %cond, ^bb1, ^bb2(%0 : i32)

^bb1: // CHECK: ^bb1:
  // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32

  // CHECK-NEXT: cf.br ^bb2(%c1_i32 : i32)
  cf.br ^bb2(%1 : i32)

^bb2(%arg : i32): // CHECK: ^bb2
  // CHECK-NEXT: %c1_i32_0 = arith.constant 1 : i32
  %2 = arith.constant 1 : i32

  // CHECK-NEXT: %1 = arith.addi %0, %c1_i32_0 : i32
  %add = arith.addi %arg, %2 : i32

  // CHECK-NEXT: return %1 : i32
  return %add : i32
}

/// The same test as above except that we are testing on a cfg embedded within
/// an operation region.
// CHECK-LABEL: func @up_propagate_region
func @up_propagate_region() -> i32 {
  // CHECK-NEXT: %0 = "foo.region"
  %0 = "foo.region"() ({
    // CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
    // CHECK-NEXT: %true = arith.constant true
    // CHECK-NEXT: cf.cond_br

    %1 = arith.constant 0 : i32
    %true = arith.constant true
    cf.cond_br %true, ^bb1, ^bb2(%1 : i32)

  ^bb1: // CHECK: ^bb1:
    // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: cf.br

    %c1_i32 = arith.constant 1 : i32
    cf.br ^bb2(%c1_i32 : i32)

  ^bb2(%arg : i32): // CHECK: ^bb2(%1: i32):
    // CHECK-NEXT: %c1_i32_0 = arith.constant 1 : i32
    // CHECK-NEXT: %2 = arith.addi %1, %c1_i32_0 : i32
    // CHECK-NEXT: "foo.yield"(%2) : (i32) -> ()

    %c1_i32_0 = arith.constant 1 : i32
    %2 = arith.addi %arg, %c1_i32_0 : i32
    "foo.yield" (%2) : (i32) -> ()
  }) : () -> (i32)
  return %0 : i32
}

/// This test checks that nested regions that are isolated from above are
/// properly handled.
// CHECK-LABEL: @nested_isolated
func @nested_isolated() -> i32 {
  // CHECK-NEXT: arith.constant 1
  %0 = arith.constant 1 : i32

  // CHECK-NEXT: @nested_func
  builtin.func @nested_func() {
    // CHECK-NEXT: arith.constant 1
    %foo = arith.constant 1 : i32
    "foo.yield"(%foo) : (i32) -> ()
  }

  // CHECK: "foo.region"
  "foo.region"() ({
    // CHECK-NEXT: arith.constant 1
    %foo = arith.constant 1 : i32
    "foo.yield"(%foo) : (i32) -> ()
  }) : () -> ()

  return %0 : i32
}

/// This test is checking that CSE gracefully handles values in graph regions
/// where the use occurs before the def, and one of the defs could be CSE'd with
/// the other.
// CHECK-LABEL: @use_before_def
func @use_before_def() {
  // CHECK-NEXT: test.graph_region
  test.graph_region {
    // CHECK-NEXT: arith.addi %c1_i32, %c1_i32_0
    %0 = arith.addi %1, %2 : i32

    // CHECK-NEXT: arith.constant 1
    // CHECK-NEXT: arith.constant 1
    %1 = arith.constant 1 : i32
    %2 = arith.constant 1 : i32

    // CHECK-NEXT: "foo.yield"(%0) : (i32) -> ()
    "foo.yield"(%0) : (i32) -> ()
  }
  return
} 
