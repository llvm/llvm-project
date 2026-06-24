// RUN: mlir-opt %s -mlir-use-nameloc-as-prefix -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-loop-unrolling='unroll-factor=2' -mlir-use-nameloc-as-prefix -split-input-file | FileCheck %s --check-prefix=CHECK-PASS-PRESERVE

// CHECK-LABEL: test_basic
func.func @test_basic() {
  %0 = memref.alloc() : memref<i32>
  // CHECK: %alice = memref.load
  %1 = memref.load %0[] : memref<i32> loc("alice")
  return
}

// -----

// CHECK-LABEL: test_repeat_namelocs
func.func @test_repeat_namelocs() {
  %0 = memref.alloc() : memref<i32>
  // CHECK: %alice = memref.load
  %1 = memref.load %0[] : memref<i32> loc("alice")
  // CHECK: %alice_0 = memref.load
  %2 = memref.load %0[] : memref<i32> loc("alice")
  return
}

// -----

// CHECK-LABEL: test_bb_args
func.func @test_bb_args1(%arg0 : memref<i32> loc("foo")) {
  // CHECK: %alice = memref.load %foo
  %1 = memref.load %arg0[] : memref<i32> loc("alice")
  return
}

// -----

func.func private @make_two_results() -> (index, index)

// CHECK-LABEL: test_multiple_results
func.func @test_multiple_results(%cond: i1) {
  // CHECK: %foo:2 = call @make_two_results
  %0:2 = call @make_two_results() : () -> (index, index) loc("foo")
  // CHECK: %bar:2 = call @make_two_results
  %1, %2 = call @make_two_results() : () -> (index, index) loc("bar")

  // CHECK: %kevin:2 = scf.while (%arg1 = %bar#0, %arg2 = %bar#0)
  %5:2 = scf.while (%arg1 = %1, %arg2 = %1) : (index, index) -> (index, index) {
    %6 = arith.cmpi slt, %arg1, %arg2 : index
    scf.condition(%6) %arg1, %arg2 : index, index
  } do {
  // CHECK: ^bb0(%alice: index, %bob: index)
  ^bb0(%arg3 : index loc("alice"), %arg4: index loc("bob")):
    %c1, %c2 = func.call @make_two_results() : () -> (index, index) loc("harriet")
    // CHECK: scf.yield %harriet#1, %harriet#1
    scf.yield %c2, %c2 : index, index
  } loc("kevin")
  return
}

// -----

#map = affine_map<(d0) -> (d0)>
#trait = {
  iterator_types = ["parallel"],
  indexing_maps = [#map, #map, #map]
}

// CHECK-LABEL: test_op_asm_interface
func.func @test_op_asm_interface(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: %c0 = arith.constant
  %0 = arith.constant 0 : index
  // CHECK: %foo = arith.constant
  %1 = arith.constant 1 : index loc("foo")

  linalg.generic #trait ins(%arg0: tensor<?xf32>) outs(%arg0, %arg1: tensor<?xf32>, tensor<?xf32>) {
    // CHECK: ^bb0(%in: f32, %out: f32, %out_0: f32)
    ^bb0(%a: f32, %b: f32, %c: f32):
      linalg.yield %a, %a : f32, f32
  } -> (tensor<?xf32>, tensor<?xf32>)

  linalg.generic #trait ins(%arg0: tensor<?xf32>) outs(%arg0, %arg1: tensor<?xf32>, tensor<?xf32>) {
    // CHECK: ^bb0(%bar: f32, %alice: f32, %steve: f32)
    ^bb0(%a: f32 loc("bar"), %b: f32 loc("alice"), %c: f32 loc("steve")):
      // CHECK: linalg.yield %alice, %steve
      linalg.yield %b, %c : f32, f32
  } -> (tensor<?xf32>, tensor<?xf32>)

  return
}

// -----

// CHECK-LABEL: test_pass
func.func @test_pass(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %arg2 = %c0 to %c4 step %c1 {
    // CHECK-PASS-PRESERVE: %foo = memref.load
    // CHECK-PASS-PRESERVE: memref.store %foo
    // CHECK-PASS-PRESERVE: %foo_1 = memref.load
    // CHECK-PASS-PRESERVE: memref.store %foo_1
    %0 = memref.load %arg0[%arg2] : memref<4xf32> loc("foo")
    memref.store %0, %arg1[%arg2] : memref<4xf32>
  }
  return
}
