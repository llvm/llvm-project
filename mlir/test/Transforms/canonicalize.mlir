// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_subi_zero
func @test_subi_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: return %c0
  %y = subi %arg0, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: func @test_subi_zero_vector
func @test_subi_zero_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  //CHECK-NEXT: %cst = constant dense<0> : vector<4xi32>
  %y = subi %arg0, %arg0 : vector<4xi32>
  // CHECK-NEXT: return %cst
  return %y: vector<4xi32>
}

// CHECK-LABEL: func @test_subi_zero_tensor
func @test_subi_zero_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  //CHECK-NEXT: %cst = constant dense<0> : tensor<4x5xi32>
  %y = subi %arg0, %arg0 : tensor<4x5xi32>
  // CHECK-NEXT: return %cst
  return %y: tensor<4x5xi32>
}

// CHECK-LABEL: func @dim
func @dim(%arg0: tensor<8x4xf32>) -> index {

  // CHECK: %c4 = constant 4 : index
  %c1 = constant 1 : index
  %0 = dim %arg0, %c1 : tensor<8x4xf32>

  // CHECK-NEXT: return %c4
  return %0 : index
}

// CHECK-LABEL: func @test_commutative
func @test_commutative(%arg0: i32) -> (i32, i32) {
  // CHECK: %c42_i32 = constant 42 : i32
  %c42_i32 = constant 42 : i32
  // CHECK-NEXT: %0 = addi %arg0, %c42_i32 : i32
  %y = addi %c42_i32, %arg0 : i32

  // This should not be swapped.
  // CHECK-NEXT: %1 = subi %c42_i32, %arg0 : i32
  %z = subi %c42_i32, %arg0 : i32

  // CHECK-NEXT: return %0, %1
  return %y, %z: i32, i32
}

// CHECK-LABEL: func @trivial_dce
func @trivial_dce(%arg0: tensor<8x4xf32>) {
  %c1 = constant 1 : index
  %0 = dim %arg0, %c1 : tensor<8x4xf32>
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @load_dce
func @load_dce(%arg0: index) {
  %c4 = constant 4 : index
  %a = alloc(%c4) : memref<?xf32>
  %2 = load %a[%arg0] : memref<?xf32>
  dealloc %a: memref<?xf32>
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @addi_zero
func @addi_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: return %arg0
  %c0_i32 = constant 0 : i32
  %y = addi %c0_i32, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: func @addi_zero_index
func @addi_zero_index(%arg0: index) -> index {
  // CHECK-NEXT: return %arg0
  %c0_index = constant 0 : index
  %y = addi %c0_index, %arg0 : index
  return %y: index
}


// CHECK-LABEL: func @addi_zero_vector
func @addi_zero_vector(%arg0: vector<4 x i32>) -> vector<4 x i32> {
  // CHECK-NEXT: return %arg0
  %c0_v4i32 = constant dense<0> : vector<4 x i32>
  %y = addi %c0_v4i32, %arg0 : vector<4 x i32>
  return %y: vector<4 x i32>
}

// CHECK-LABEL: func @addi_zero_tensor
func @addi_zero_tensor(%arg0: tensor<4 x 5 x i32>) -> tensor<4 x 5 x i32> {
  // CHECK-NEXT: return %arg0
  %c0_t45i32 = constant dense<0> : tensor<4 x 5 x i32>
  %y = addi %arg0, %c0_t45i32 : tensor<4 x 5 x i32>
  return %y: tensor<4 x 5 x i32>
}

// CHECK-LABEL: func @muli_zero
func @muli_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  %c0_i32 = constant 0 : i32

  %y = muli %c0_i32, %arg0 : i32

  // CHECK-NEXT: return %c0_i32
  return %y: i32
}

// CHECK-LABEL: func @muli_zero_index
func @muli_zero_index(%arg0: index) -> index {
  // CHECK-NEXT: %[[CST:.*]] = constant 0 : index
  %c0_index = constant 0 : index

  %y = muli %c0_index, %arg0 : index

  // CHECK-NEXT: return %[[CST]]
  return %y: index
}

// CHECK-LABEL: func @muli_zero_vector
func @muli_zero_vector(%arg0: vector<4 x i32>) -> vector<4 x i32> {
  // CHECK-NEXT: %cst = constant dense<0> : vector<4xi32>
  %cst = constant dense<0> : vector<4 x i32>

  %y = muli %cst, %arg0 : vector<4 x i32>

  // CHECK-NEXT: return %cst
  return %y: vector<4 x i32>
}

// CHECK-LABEL: func @muli_zero_tensor
func @muli_zero_tensor(%arg0: tensor<4 x 5 x i32>) -> tensor<4 x 5 x i32> {
  // CHECK-NEXT: %cst = constant dense<0> : tensor<4x5xi32>
  %cst = constant dense<0> : tensor<4 x 5 x i32>

  %y = muli %arg0, %cst : tensor<4 x 5 x i32>

  // CHECK-NEXT: return %cst
  return %y: tensor<4 x 5 x i32>
}

// CHECK-LABEL: func @muli_one
func @muli_one(%arg0: i32) -> i32 {
  // CHECK-NEXT: return %arg0
  %c0_i32 = constant 1 : i32
  %y = muli %c0_i32, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: func @muli_one_index
func @muli_one_index(%arg0: index) -> index {
  // CHECK-NEXT: return %arg0
  %c0_index = constant 1 : index
  %y = muli %c0_index, %arg0 : index
  return %y: index
}

// CHECK-LABEL: func @muli_one_vector
func @muli_one_vector(%arg0: vector<4 x i32>) -> vector<4 x i32> {
  // CHECK-NEXT: return %arg0
  %c1_v4i32 = constant dense<1> : vector<4 x i32>
  %y = muli %c1_v4i32, %arg0 : vector<4 x i32>
  return %y: vector<4 x i32>
}

// CHECK-LABEL: func @muli_one_tensor
func @muli_one_tensor(%arg0: tensor<4 x 5 x i32>) -> tensor<4 x 5 x i32> {
  // CHECK-NEXT: return %arg0
  %c1_t45i32 = constant dense<1> : tensor<4 x 5 x i32>
  %y = muli %arg0, %c1_t45i32 : tensor<4 x 5 x i32>
  return %y: tensor<4 x 5 x i32>
}

//CHECK-LABEL: func @and_self
func @and_self(%arg0: i32) -> i32 {
  //CHECK-NEXT: return %arg0
  %1 = and %arg0, %arg0 : i32
  return %1 : i32
}

//CHECK-LABEL: func @and_self_vector
func @and_self_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  //CHECK-NEXT: return %arg0
  %1 = and %arg0, %arg0 : vector<4xi32>
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @and_self_tensor
func @and_self_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  //CHECK-NEXT: return %arg0
  %1 = and %arg0, %arg0 : tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

//CHECK-LABEL: func @and_zero
func @and_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  %c0_i32 = constant 0 : i32
  // CHECK-NEXT: return %c0_i32
  %1 = and %arg0, %c0_i32 : i32
  return %1 : i32
}

//CHECK-LABEL: func @and_zero_index
func @and_zero_index(%arg0: index) -> index {
  // CHECK-NEXT: %[[CST:.*]] = constant 0 : index
  %c0_index = constant 0 : index
  // CHECK-NEXT: return %[[CST]]
  %1 = and %arg0, %c0_index : index
  return %1 : index
}

//CHECK-LABEL: func @and_zero_vector
func @and_zero_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  // CHECK-NEXT: %cst = constant dense<0> : vector<4xi32>
  %cst = constant dense<0> : vector<4xi32>
  // CHECK-NEXT: return %cst
  %1 = and %arg0, %cst : vector<4xi32>
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @and_zero_tensor
func @and_zero_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  // CHECK-NEXT: %cst = constant dense<0> : tensor<4x5xi32>
  %cst = constant dense<0> : tensor<4x5xi32>
  // CHECK-NEXT: return %cst
  %1 = and %arg0, %cst : tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

//CHECK-LABEL: func @or_self
func @or_self(%arg0: i32) -> i32 {
  //CHECK-NEXT: return %arg0
  %1 = or %arg0, %arg0 : i32
  return %1 : i32
}

//CHECK-LABEL: func @or_self_vector
func @or_self_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  //CHECK-NEXT: return %arg0
  %1 = or %arg0, %arg0 : vector<4xi32>
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @or_self_tensor
func @or_self_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  //CHECK-NEXT: return %arg0
  %1 = or %arg0, %arg0 : tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

//CHECK-LABEL: func @or_zero
func @or_zero(%arg0: i32) -> i32 {
  %c0_i32 = constant 0 : i32
  // CHECK-NEXT: return %arg0
  %1 = or %arg0, %c0_i32 : i32
  return %1 : i32
}

//CHECK-LABEL: func @or_zero_index
func @or_zero_index(%arg0: index) -> index {
  %c0_index = constant 0 : index
  // CHECK-NEXT: return %arg0
  %1 = or %arg0, %c0_index : index
  return %1 : index
}

//CHECK-LABEL: func @or_zero_vector
func @or_zero_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  // CHECK-NEXT: return %arg0
  %cst = constant dense<0> : vector<4xi32>
  %1 = or %arg0, %cst : vector<4xi32>
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @or_zero_tensor
func @or_zero_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  // CHECK-NEXT: return %arg0
  %cst = constant dense<0> : tensor<4x5xi32>
  %1 = or %arg0, %cst : tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

//CHECK-LABEL: func @xor_self
func @xor_self(%arg0: i32) -> i32 {
  //CHECK-NEXT: %c0_i32 = constant 0
  %1 = xor %arg0, %arg0 : i32
  //CHECK-NEXT: return %c0_i32
  return %1 : i32
}

//CHECK-LABEL: func @xor_self_vector
func @xor_self_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  //CHECK-NEXT: %cst = constant dense<0> : vector<4xi32>
  %1 = xor %arg0, %arg0 : vector<4xi32>
  //CHECK-NEXT: return %cst
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @xor_self_tensor
func @xor_self_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  //CHECK-NEXT: %cst = constant dense<0> : tensor<4x5xi32>
  %1 = xor %arg0, %arg0 : tensor<4x5xi32>
  //CHECK-NEXT: return %cst
  return %1 : tensor<4x5xi32>
}

// CHECK-LABEL: func @memref_cast_folding
func @memref_cast_folding(%arg0: memref<4 x f32>, %arg1: f32) -> (f32, f32) {
  %0 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>
  // CHECK-NEXT: %c0 = constant 0 : index
  %c0 = constant 0 : index
  %dim = dim %0, %c0 : memref<? x f32>

  // CHECK-NEXT: affine.load %arg0[3]
  %1 = affine.load %0[%dim - 1] : memref<?xf32>

  // CHECK-NEXT: store %arg1, %arg0[%c0] : memref<4xf32>
  store %arg1, %0[%c0] : memref<?xf32>

  // CHECK-NEXT: %{{.*}} = load %arg0[%c0] : memref<4xf32>
  %2 = load %0[%c0] : memref<?xf32>

  // CHECK-NEXT: dealloc %arg0 : memref<4xf32>
  dealloc %0: memref<?xf32>

  // CHECK-NEXT: return %{{.*}}
  return %1, %2 : f32, f32
}

// CHECK-LABEL: @fold_memref_cast_in_memref_cast
// CHECK-SAME: (%[[ARG0:.*]]: memref<42x42xf64>)
func @fold_memref_cast_in_memref_cast(%0: memref<42x42xf64>) {
  // CHECK: %[[folded:.*]] = memref_cast %[[ARG0]] : memref<42x42xf64> to memref<?x?xf64>
  %4 = memref_cast %0 : memref<42x42xf64> to memref<?x42xf64>
  // CHECK-NOT: memref_cast
  %5 = memref_cast %4 : memref<?x42xf64> to memref<?x?xf64>
  // CHECK: "test.user"(%[[folded]])
  "test.user"(%5) : (memref<?x?xf64>) -> ()
  return
}

// CHECK-LABEL: @fold_memref_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: memref<42x42xf64>)
func @fold_memref_cast_chain(%0: memref<42x42xf64>) {
  // CHECK-NOT: memref_cast
  %4 = memref_cast %0 : memref<42x42xf64> to memref<?x42xf64>
  %5 = memref_cast %4 : memref<?x42xf64> to memref<42x42xf64>
  // CHECK: "test.user"(%[[ARG0]])
  "test.user"(%5) : (memref<42x42xf64>) -> ()
  return
}

// CHECK-LABEL: func @alloc_const_fold
func @alloc_const_fold() -> memref<?xf32> {
  // CHECK-NEXT: %0 = alloc() : memref<4xf32>
  %c4 = constant 4 : index
  %a = alloc(%c4) : memref<?xf32>

  // CHECK-NEXT: %1 = memref_cast %0 : memref<4xf32> to memref<?xf32>
  // CHECK-NEXT: return %1 : memref<?xf32>
  return %a : memref<?xf32>
}

// CHECK-LABEL: func @dead_alloc_fold
func @dead_alloc_fold() {
  // CHECK-NEXT: return
  %c4 = constant 4 : index
  %a = alloc(%c4) : memref<?xf32>
  return
}

// CHECK-LABEL: func @dead_dealloc_fold
func @dead_dealloc_fold() {
  // CHECK-NEXT: return
  %a = alloc() : memref<4xf32>
  dealloc %a: memref<4xf32>
  return
}

// CHECK-LABEL: func @dead_dealloc_fold_multi_use
func @dead_dealloc_fold_multi_use(%cond : i1) {
  // CHECK-NEXT: return
  %a = alloc() : memref<4xf32>
  cond_br %cond, ^bb1, ^bb2

^bb1:
  dealloc %a: memref<4xf32>
  return

^bb2:
  dealloc %a: memref<4xf32>
  return
}

// CHECK-LABEL: func @dead_block_elim
func @dead_block_elim() {
  // CHECK-NOT: ^bb
  func @nested() {
    return

  ^bb1:
    return
  }
  return

^bb1:
  return
}

// CHECK-LABEL: func @dyn_shape_fold(%arg0: index, %arg1: index)
func @dyn_shape_fold(%L : index, %M : index) -> (memref<? x ? x i32>, memref<? x ? x f32>) {
  // CHECK: %c0 = constant 0 : index
  %zero = constant 0 : index
  // The constants below disappear after they propagate into shapes.
  %nine = constant 9 : index
  %N = constant 1024 : index
  %K = constant 512 : index

  // CHECK-NEXT: alloc(%arg0) : memref<?x1024xf32>
  %a = alloc(%L, %N) : memref<? x ? x f32>

  // CHECK-NEXT: alloc(%arg1) : memref<4x1024x8x512x?xf32>
  %b = alloc(%N, %K, %M) : memref<4 x ? x 8 x ? x ? x f32>

  // CHECK-NEXT: alloc() : memref<512x1024xi32>
  %c = alloc(%K, %N) : memref<? x ? x i32>

  // CHECK: alloc() : memref<9x9xf32>
  %d = alloc(%nine, %nine) : memref<? x ? x f32>

  // CHECK: alloca(%arg1) : memref<4x1024x8x512x?xf32>
  %e = alloca(%N, %K, %M) : memref<4 x ? x 8 x ? x ? x f32>

  // CHECK: affine.for
  affine.for %i = 0 to %L {
    // CHECK-NEXT: affine.for
    affine.for %j = 0 to 10 {
      // CHECK-NEXT: load %0[%arg2, %arg3] : memref<?x1024xf32>
      // CHECK-NEXT: store %{{.*}}, %1[%c0, %c0, %arg2, %arg3, %c0] : memref<4x1024x8x512x?xf32>
      %v = load %a[%i, %j] : memref<?x?xf32>
      store %v, %b[%zero, %zero, %i, %j, %zero] : memref<4x?x8x?x?xf32>
    }
  }

  return %c, %d : memref<? x ? x i32>, memref<? x ? x f32>
}

#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map2 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s2 + d1 * s1 + d2 + s0)>
#map3 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

// CHECK-LABEL: func @dim_op_fold(
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: index
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: index
// CHECK-SAME: %[[ARG2:[a-z0-9]*]]: index
// CHECK-SAME: %[[BUF:[a-z0-9]*]]: memref<?xi8>
func @dim_op_fold(%arg0: index, %arg1: index, %arg2: index, %BUF: memref<?xi8>, %M : index, %N : index, %K : index) {
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
  %1 = alloc(%arg1, %arg2) : memref<?x8x?xf32>
  %2 = dim %1, %c2 : memref<?x8x?xf32>
  affine.for %arg3 = 0 to %2 {
    %3 = alloc(%arg0) : memref<?xi8>
    %ub = dim %3, %c0 : memref<?xi8>
    affine.for %arg4 = 0 to %ub {
      %s = dim %0, %c0 : memref<?x?xf32>
      %v = std.view %3[%c0][%arg4, %s] : memref<?xi8> to memref<?x?xf32>
      %sv = subview %0[%c0, %c0][%s,%arg4][%c1,%c1] : memref<?x?xf32> to memref<?x?xf32, #map1>
      %l = dim %v, %c1 : memref<?x?xf32>
      %u = dim %sv, %c0 : memref<?x?xf32, #map1>
      affine.for %arg5 = %l to %u {
        "foo"() : () -> ()
      }
      %sv2 = subview %0[0, 0][17, %arg4][1, 1] : memref<?x?xf32> to memref<17x?xf32, #map3>
      %l2 = dim %v, %c1 : memref<?x?xf32>
      %u2 = dim %sv2, %c1 : memref<17x?xf32, #map3>
      scf.for %arg5 = %l2 to %u2 step %c1 {
        "foo"() : () -> ()
      }
    }
  }
  //      CHECK: affine.for %[[I:.*]] = 0 to %[[ARG2]] {
  // CHECK-NEXT:   affine.for %[[J:.*]] = 0 to %[[ARG0]] {
  // CHECK-NEXT:     affine.for %[[K:.*]] = %[[ARG0]] to %[[ARG0]] {
  // CHECK-NEXT:       "foo"() : () -> ()
  // CHECK-NEXT:     }
  // CHECK-NEXT:     scf.for %[[KK:.*]] = %[[ARG0]] to %[[J]] step %{{.*}} {
  // CHECK-NEXT:       "foo"() : () -> ()
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  %A = view %BUF[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32>
  %B = view %BUF[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32>
  %C = view %BUF[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>

  %M_ = dim %A, %c0 : memref<?x?xf32>
  %K_ = dim %A, %c1 : memref<?x?xf32>
  %N_ = dim %C, %c1 : memref<?x?xf32>
  scf.for %i = %c0 to %M_ step %c1 {
    scf.for %j = %c0 to %N_ step %c1 {
      scf.for %k = %c0 to %K_ step %c1 {
      }
    }
  }
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @merge_constants
func @merge_constants() -> (index, index) {
  // CHECK-NEXT: %c42 = constant 42 : index
  %0 = constant 42 : index
  %1 = constant 42 : index
  // CHECK-NEXT: return %c42, %c42
  return %0, %1: index, index
}

// CHECK-LABEL: func @hoist_constant
func @hoist_constant(%arg0: memref<8xi32>) {
  // CHECK-NEXT: %c42_i32 = constant 42 : i32
  // CHECK-NEXT: affine.for %arg1 = 0 to 8 {
  affine.for %arg1 = 0 to 8 {
    // CHECK-NEXT: store %c42_i32, %arg0[%arg1]
    %c42_i32 = constant 42 : i32
    store %c42_i32, %arg0[%arg1] : memref<8xi32>
  }
  return
}

// CHECK-LABEL: func @const_fold_propagate
func @const_fold_propagate() -> memref<?x?xf32> {
  %VT_i = constant 512 : index

  %VT_i_s = affine.apply affine_map<(d0) -> (d0 floordiv  8)> (%VT_i)
  %VT_k_l = affine.apply affine_map<(d0) -> (d0 floordiv  16)> (%VT_i)

  // CHECK: = alloc() : memref<64x32xf32>
  %Av = alloc(%VT_i_s, %VT_k_l) : memref<?x?xf32>
  return %Av : memref<?x?xf32>
}

// CHECK-LABEL: func @indirect_call_folding
func @indirect_target() {
  return
}

func @indirect_call_folding() {
  // CHECK-NEXT: call @indirect_target() : () -> ()
  // CHECK-NEXT: return
  %indirect_fn = constant @indirect_target : () -> ()
  call_indirect %indirect_fn() : () -> ()
  return
}

//
// IMPORTANT NOTE: the operations in this test are exactly those produced by
// lowering affine.apply affine_map<(i) -> (i mod 42)> to standard operations.  Please only
// change these operations together with the affine lowering pass tests.
//
// CHECK-LABEL: @lowered_affine_mod
func @lowered_affine_mod() -> (index, index) {
// CHECK-NEXT: {{.*}} = constant 41 : index
  %c-43 = constant -43 : index
  %c42 = constant 42 : index
  %0 = remi_signed %c-43, %c42 : index
  %c0 = constant 0 : index
  %1 = cmpi "slt", %0, %c0 : index
  %2 = addi %0, %c42 : index
  %3 = select %1, %2, %0 : index
// CHECK-NEXT: {{.*}} = constant 1 : index
  %c43 = constant 43 : index
  %c42_0 = constant 42 : index
  %4 = remi_signed %c43, %c42_0 : index
  %c0_1 = constant 0 : index
  %5 = cmpi "slt", %4, %c0_1 : index
  %6 = addi %4, %c42_0 : index
  %7 = select %5, %6, %4 : index
  return %3, %7 : index, index
}

//
// IMPORTANT NOTE: the operations in this test are exactly those produced by
// lowering affine.apply affine_map<(i) -> (i mod 42)> to standard operations.  Please only
// change these operations together with the affine lowering pass tests.
//
// CHECK-LABEL: func @lowered_affine_floordiv
func @lowered_affine_floordiv() -> (index, index) {
// CHECK-NEXT: %c-2 = constant -2 : index
  %c-43 = constant -43 : index
  %c42 = constant 42 : index
  %c0 = constant 0 : index
  %c-1 = constant -1 : index
  %0 = cmpi "slt", %c-43, %c0 : index
  %1 = subi %c-1, %c-43 : index
  %2 = select %0, %1, %c-43 : index
  %3 = divi_signed %2, %c42 : index
  %4 = subi %c-1, %3 : index
  %5 = select %0, %4, %3 : index
// CHECK-NEXT: %c1 = constant 1 : index
  %c43 = constant 43 : index
  %c42_0 = constant 42 : index
  %c0_1 = constant 0 : index
  %c-1_2 = constant -1 : index
  %6 = cmpi "slt", %c43, %c0_1 : index
  %7 = subi %c-1_2, %c43 : index
  %8 = select %6, %7, %c43 : index
  %9 = divi_signed %8, %c42_0 : index
  %10 = subi %c-1_2, %9 : index
  %11 = select %6, %10, %9 : index
  return %5, %11 : index, index
}

//
// IMPORTANT NOTE: the operations in this test are exactly those produced by
// lowering affine.apply affine_map<(i) -> (i mod 42)> to standard operations.  Please only
// change these operations together with the affine lowering pass tests.
//
// CHECK-LABEL: func @lowered_affine_ceildiv
func @lowered_affine_ceildiv() -> (index, index) {
// CHECK-NEXT:  %c-1 = constant -1 : index
  %c-43 = constant -43 : index
  %c42 = constant 42 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = cmpi "sle", %c-43, %c0 : index
  %1 = subi %c0, %c-43 : index
  %2 = subi %c-43, %c1 : index
  %3 = select %0, %1, %2 : index
  %4 = divi_signed %3, %c42 : index
  %5 = subi %c0, %4 : index
  %6 = addi %4, %c1 : index
  %7 = select %0, %5, %6 : index
// CHECK-NEXT:  %c2 = constant 2 : index
  %c43 = constant 43 : index
  %c42_0 = constant 42 : index
  %c0_1 = constant 0 : index
  %c1_2 = constant 1 : index
  %8 = cmpi "sle", %c43, %c0_1 : index
  %9 = subi %c0_1, %c43 : index
  %10 = subi %c43, %c1_2 : index
  %11 = select %8, %9, %10 : index
  %12 = divi_signed %11, %c42_0 : index
  %13 = subi %c0_1, %12 : index
  %14 = addi %12, %c1_2 : index
  %15 = select %8, %13, %14 : index
  return %7, %15 : index, index
}

// Checks that NOP casts are removed.
// CHECK-LABEL: cast_values
func @cast_values(%arg0: tensor<*xi32>, %arg1: memref<?xi32>) -> (tensor<2xi32>, memref<2xi32>) {

  // NOP casts
  %0 = tensor_cast %arg0 : tensor<*xi32> to tensor<*xi32>
  %1 = memref_cast %arg1 : memref<?xi32> to memref<?xi32>

  // CHECK-NEXT: %0 = tensor_cast %arg0 : tensor<*xi32> to tensor<2xi32>
  // CHECK-NEXT: %1 = memref_cast %arg1 : memref<?xi32> to memref<2xi32>
  %2 = tensor_cast %0 : tensor<*xi32> to tensor<2xi32>
  %3 = memref_cast %1 : memref<?xi32> to memref<2xi32>

  // NOP casts
  %4 = tensor_cast %2 : tensor<2xi32> to tensor<2xi32>
  %5 = memref_cast %3 : memref<2xi32> to memref<2xi32>

  // CHECK-NEXT: return %0, %1 : tensor<2xi32>, memref<2xi32>
  return %4, %5 : tensor<2xi32>, memref<2xi32>
}

// -----

// CHECK-LABEL: func @view
func @view(%arg0 : index) -> (f32, f32, f32, f32) {
  // CHECK: %[[C15:.*]] = constant 15 : index
  // CHECK: %[[ALLOC_MEM:.*]] = alloc() : memref<2048xi8>
  %0 = alloc() : memref<2048xi8>
  %c0 = constant 0 : index
  %c7 = constant 7 : index
  %c11 = constant 11 : index
  %c15 = constant 15 : index

  // Test: fold constant sizes.
  // CHECK: std.view %[[ALLOC_MEM]][%[[C15]]][] : memref<2048xi8> to memref<7x11xf32>
  %1 = view %0[%c15][%c7, %c11] : memref<2048xi8> to memref<?x?xf32>
  %r0 = load %1[%c0, %c0] : memref<?x?xf32>

  // Test: fold one constant size.
  // CHECK: std.view %[[ALLOC_MEM]][%[[C15]]][%arg0, %arg0] : memref<2048xi8> to memref<?x?x7xf32>
  %2 = view %0[%c15][%arg0, %arg0, %c7] : memref<2048xi8> to memref<?x?x?xf32>
  %r1 = load %2[%c0, %c0, %c0] : memref<?x?x?xf32>

  // Test: preserve an existing static size.
  // CHECK: std.view %[[ALLOC_MEM]][%[[C15]]][] : memref<2048xi8> to memref<7x4xf32>
  %3 = view %0[%c15][%c7] : memref<2048xi8> to memref<?x4xf32>
  %r2 = load %3[%c0, %c0] : memref<?x4xf32>

  // Test: folding static alloc and memref_cast into a view.
  // CHECK: std.view %[[ALLOC_MEM]][%[[C15]]][] : memref<2048xi8> to memref<15x7xf32>
  %4 = memref_cast %0 : memref<2048xi8> to memref<?xi8>
  %5 = view %4[%c15][%c15, %c7] : memref<?xi8> to memref<?x?xf32>
  %r3 = load %5[%c0, %c0] : memref<?x?xf32>
  return %r0, %r1, %r2, %r3 : f32, f32, f32, f32
}

// -----

// CHECK-DAG: #[[$BASE_MAP0:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>
// CHECK-DAG: #[[$SUBVIEW_MAP0:map[0-9]+]] = affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + s0 + d1 * 4 + d2)>
// CHECK-DAG: #[[$SUBVIEW_MAP1:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 79)>
// CHECK-DAG: #[[$SUBVIEW_MAP2:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0 * 128 + d1 * 28 + d2 * 11)>
// CHECK-DAG: #[[$SUBVIEW_MAP3:map[0-9]+]] = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>
// CHECK-DAG: #[[$SUBVIEW_MAP4:map[0-9]+]] = affine_map<(d0, d1, d2)[s0] -> (d0 * 128 + s0 + d1 * 28 + d2 * 11)>
// CHECK-DAG: #[[$SUBVIEW_MAP5:map[0-9]+]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s0 + d1 * s1 + d2 * s2 + 79)>
// CHECK-DAG: #[[$SUBVIEW_MAP6:map[0-9]+]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2 * 2)>
// CHECK-DAG: #[[$SUBVIEW_MAP7:map[0-9]+]] = affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>
// CHECK-DAG: #[[$SUBVIEW_MAP8:map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 4 + d1 + 12)>


// CHECK-LABEL: func @subview
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func @subview(%arg0 : index, %arg1 : index) -> (index, index) {
  // CHECK: %[[C0:.*]] = constant 0 : index
  %c0 = constant 0 : index
  // CHECK-NOT: constant 1 : index
  %c1 = constant 1 : index
  // CHECK-NOT: constant 2 : index
  %c2 = constant 2 : index
  // Folded but reappears after subview folding into dim.
  // CHECK: %[[C7:.*]] = constant 7 : index
  %c7 = constant 7 : index
  // Folded but reappears after subview folding into dim.
  // CHECK: %[[C11:.*]] = constant 11 : index
  %c11 = constant 11 : index
  // CHECK-NOT: constant 15 : index
  %c15 = constant 15 : index

  // CHECK: %[[ALLOC0:.*]] = alloc()
  %0 = alloc() : memref<8x16x4xf32, offset : 0, strides : [64, 4, 1]>

  // Test: subview with constant base memref and constant operands is folded.
  // Note that the subview uses the base memrefs layout map because it used
  // zero offset and unit stride arguments.
  // CHECK: subview %[[ALLOC0]][0, 0, 0] [7, 11, 2] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]>
  // CHECK-SAME: to memref<7x11x2xf32, #[[$BASE_MAP0]]>
  %1 = subview %0[%c0, %c0, %c0] [%c7, %c11, %c2] [%c1, %c1, %c1]
    : memref<8x16x4xf32, offset : 0, strides : [64, 4, 1]> to
      memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>
  %v0 = load %1[%c0, %c0, %c0] : memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>

  // Test: subview with one dynamic operand can also be folded.
  // CHECK: subview %[[ALLOC0]][0, %[[ARG0]], 0] [7, 11, 15] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]>
  // CHECK-SAME: to memref<7x11x15xf32, #[[$SUBVIEW_MAP0]]>
  %2 = subview %0[%c0, %arg0, %c0] [%c7, %c11, %c15] [%c1, %c1, %c1]
    : memref<8x16x4xf32, offset : 0, strides : [64, 4, 1]> to
      memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>
  store %v0, %2[%c0, %c0, %c0] : memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>

  // CHECK: %[[ALLOC1:.*]] = alloc(%[[ARG0]])
  %3 = alloc(%arg0) : memref<?x16x4xf32, offset : 0, strides : [64, 4, 1]>
  // Test: subview with constant operands but dynamic base memref is folded as long as the strides and offset of the base memref are static.
  // CHECK: subview %[[ALLOC1]][0, 0, 0] [7, 11, 15] [1, 1, 1] :
  // CHECK-SAME: memref<?x16x4xf32, #[[$BASE_MAP0]]>
  // CHECK-SAME: to memref<7x11x15xf32, #[[$BASE_MAP0]]>
  %4 = subview %3[%c0, %c0, %c0] [%c7, %c11, %c15] [%c1, %c1, %c1]
    : memref<?x16x4xf32, offset : 0, strides : [64, 4, 1]> to
      memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>
  store %v0, %4[%c0, %c0, %c0] : memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>

  // Test: subview offset operands are folded correctly w.r.t. base strides.
  // CHECK: subview %[[ALLOC0]][1, 2, 7] [7, 11, 2] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]> to
  // CHECK-SAME: memref<7x11x2xf32, #[[$SUBVIEW_MAP1]]>
  %5 = subview %0[%c1, %c2, %c7] [%c7, %c11, %c2] [%c1, %c1, %c1]
    : memref<8x16x4xf32, offset : 0, strides : [64, 4, 1]> to
      memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>
  store %v0, %5[%c0, %c0, %c0] : memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>

  // Test: subview stride operands are folded correctly w.r.t. base strides.
  // CHECK: subview %[[ALLOC0]][0, 0, 0] [7, 11, 2] [2, 7, 11] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]>
  // CHECK-SAME: to memref<7x11x2xf32, #[[$SUBVIEW_MAP2]]>
  %6 = subview %0[%c0, %c0, %c0] [%c7, %c11, %c2] [%c2, %c7, %c11]
    : memref<8x16x4xf32, offset : 0, strides : [64, 4, 1]> to
      memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>
  store %v0, %6[%c0, %c0, %c0] : memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>

  // Test: subview shape are folded, but offsets and strides are not even if base memref is static
  // CHECK: subview %[[ALLOC0]][%[[ARG0]], %[[ARG0]], %[[ARG0]]] [7, 11, 2] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]> to
  // CHECK-SAME: memref<7x11x2xf32, #[[$SUBVIEW_MAP3]]>
  %10 = subview %0[%arg0, %arg0, %arg0] [%c7, %c11, %c2] [%arg1, %arg1, %arg1] :
    memref<8x16x4xf32, offset:0, strides:[64, 4, 1]> to
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>
  store %v0, %10[%arg1, %arg1, %arg1] :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>

  // Test: subview strides are folded, but offsets and shape are not even if base memref is static
  // CHECK: subview %[[ALLOC0]][%[[ARG0]], %[[ARG0]], %[[ARG0]]] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] [2, 7, 11] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]> to
  // CHECK-SAME: memref<?x?x?xf32, #[[$SUBVIEW_MAP4]]
  %11 = subview %0[%arg0, %arg0, %arg0] [%arg1, %arg1, %arg1] [%c2, %c7, %c11] :
    memref<8x16x4xf32, offset:0, strides:[64, 4, 1]> to
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>
  store %v0, %11[%arg0, %arg0, %arg0] :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>

  // Test: subview offsets are folded, but strides and shape are not even if base memref is static
  // CHECK: subview %[[ALLOC0]][1, 2, 7] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] [%[[ARG0]], %[[ARG0]], %[[ARG0]]] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]> to
  // CHECK-SAME: memref<?x?x?xf32, #[[$SUBVIEW_MAP5]]
  %13 = subview %0[%c1, %c2, %c7] [%arg1, %arg1, %arg1] [%arg0, %arg0, %arg0] :
    memref<8x16x4xf32, offset:0, strides:[64, 4, 1]> to
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>
  store %v0, %13[%arg1, %arg1, %arg1] :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>

  // CHECK: %[[ALLOC2:.*]] = alloc(%[[ARG0]], %[[ARG0]], %[[ARG1]])
  %14 = alloc(%arg0, %arg0, %arg1) : memref<?x?x?xf32>
  // Test: subview shape are folded, even if base memref is not static
  // CHECK: subview %[[ALLOC2]][%[[ARG0]], %[[ARG0]], %[[ARG0]]] [7, 11, 2] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] :
  // CHECK-SAME: memref<?x?x?xf32> to
  // CHECK-SAME: memref<7x11x2xf32, #[[$SUBVIEW_MAP3]]>
  %15 = subview %14[%arg0, %arg0, %arg0] [%c7, %c11, %c2] [%arg1, %arg1, %arg1] :
    memref<?x?x?xf32> to
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>
  store %v0, %15[%arg1, %arg1, %arg1] : memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>

  // TEST: subview strides are folded, in the type only the most minor stride is folded.
  // CHECK: subview %[[ALLOC2]][%[[ARG0]], %[[ARG0]], %[[ARG0]]] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] [2, 2, 2] :
  // CHECK-SAME: memref<?x?x?xf32> to
  // CHECK-SAME: memref<?x?x?xf32, #[[$SUBVIEW_MAP6]]
  %16 = subview %14[%arg0, %arg0, %arg0] [%arg1, %arg1, %arg1] [%c2, %c2, %c2] :
    memref<?x?x?xf32> to
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>
  store %v0, %16[%arg0, %arg0, %arg0] : memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>

  // TEST: subview offsets are folded but the type offset remains dynamic, when the base memref is not static
  // CHECK: subview %[[ALLOC2]][1, 1, 1] [%[[ARG0]], %[[ARG0]], %[[ARG0]]] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] :
  // CHECK-SAME: memref<?x?x?xf32> to
  // CHECK-SAME: memref<?x?x?xf32, #[[$SUBVIEW_MAP3]]
  %17 = subview %14[%c1, %c1, %c1] [%arg0, %arg0, %arg0] [%arg1, %arg1, %arg1] :
    memref<?x?x?xf32> to
    memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>
  store %v0, %17[%arg0, %arg0, %arg0] : memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>

  // CHECK: %[[ALLOC3:.*]] = alloc() : memref<12x4xf32>
  %18 = alloc() : memref<12x4xf32>
  %c4 = constant 4 : index

  // TEST: subview strides are maintained when sizes are folded
  // CHECK: subview %[[ALLOC3]][%arg1, %arg1] [2, 4] [1, 1] :
  // CHECK-SAME: memref<12x4xf32> to
  // CHECK-SAME: memref<2x4xf32, #[[$SUBVIEW_MAP7]]>
  %19 = subview %18[%arg1, %arg1] [%c2, %c4] [1, 1] :
    memref<12x4xf32> to
    memref<?x?xf32, offset: ?, strides:[4, 1]>
  store %v0, %19[%arg1, %arg1] : memref<?x?xf32, offset: ?, strides:[4, 1]>

  // TEST: subview strides and sizes are maintained when offsets are folded
  // CHECK: subview %[[ALLOC3]][2, 4] [12, 4] [1, 1] :
  // CHECK-SAME: memref<12x4xf32> to
  // CHECK-SAME: memref<12x4xf32, #[[$SUBVIEW_MAP8]]>
  %20 = subview %18[%c2, %c4] [12, 4] [1, 1] :
    memref<12x4xf32> to
    memref<12x4xf32, offset: ?, strides:[4, 1]>
  store %v0, %20[%arg1, %arg1] : memref<12x4xf32, offset: ?, strides:[4, 1]>

  // Test: dim on subview is rewritten to size operand.
  %7 = dim %4, %c0 : memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>
  %8 = dim %4, %c1 : memref<?x?x?xf32, offset : ?, strides : [?, ?, ?]>

  // CHECK: return %[[C7]], %[[C11]]
  return %7, %8 : index, index
}

// CHECK-LABEL: func @index_cast
// CHECK-SAME: %[[ARG_0:arg[0-9]+]]: i16
func @index_cast(%arg0: i16) -> (i16) {
  %11 = index_cast %arg0 : i16 to index
  %12 = index_cast %11 : index to i16
  // CHECK: return %[[ARG_0]] : i16
  return %12 : i16
}

// CHECK-LABEL: func @index_cast_fold
func @index_cast_fold() -> (i16, index) {
  %c4 = constant 4 : index
  %1 = index_cast %c4 : index to i16
  %c4_i16 = constant 4 : i16
  %2 = index_cast %c4_i16 : i16 to index
  // CHECK: %[[C4_I16:.*]] = constant 4 : i16
  // CHECK: %[[C4:.*]] = constant 4 : index
  // CHECK: return %[[C4_I16]], %[[C4]] : i16, index
  return %1, %2 : i16, index
}

// CHECK-LABEL: func @remove_dead_else
func @remove_dead_else(%M : memref<100 x i32>) {
  affine.for %i = 0 to 100 {
    affine.load %M[%i] : memref<100xi32>
    affine.if affine_set<(d0) : (d0 - 2 >= 0)>(%i) {
      affine.for %j = 0 to 100 {
        %1 = affine.load %M[%j] : memref<100xi32>
        "prevent.dce"(%1) : (i32) -> ()
      }
    } else {
      // Nothing
    }
    affine.load %M[%i] : memref<100xi32>
  }
  return
}
// CHECK:      affine.if
// CHECK-NEXT:   affine.for
// CHECK-NEXT:     affine.load
// CHECK-NEXT:     "prevent.dce"
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// CHECK-LABEL: func @divi_signed_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func @divi_signed_by_one(%arg0: i32) -> (i32) {
  %c1 = constant 1 : i32
  %res = divi_signed %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @divi_unsigned_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func @divi_unsigned_by_one(%arg0: i32) -> (i32) {
  %c1 = constant 1 : i32
  %res = divi_unsigned %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @tensor_divi_signed_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func @tensor_divi_signed_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = constant dense<1> : tensor<4x5xi32>
  %res = divi_signed %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// CHECK-LABEL: func @tensor_divi_unsigned_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func @tensor_divi_unsigned_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = constant dense<1> : tensor<4x5xi32>
  %res = divi_unsigned %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// -----

// CHECK-LABEL: func @floordivi_signed_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func @floordivi_signed_by_one(%arg0: i32) -> (i32) {
  %c1 = constant 1 : i32
  %res = floordivi_signed %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @tensor_floordivi_signed_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func @tensor_floordivi_signed_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = constant dense<1> : tensor<4x5xi32>
  %res = floordivi_signed %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// -----

// CHECK-LABEL: func @ceildivi_signed_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func @ceildivi_signed_by_one(%arg0: i32) -> (i32) {
  %c1 = constant 1 : i32
  %res = ceildivi_signed %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @tensor_ceildivi_signed_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func @tensor_ceildivi_signed_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = constant dense<1> : tensor<4x5xi32>
  %res = ceildivi_signed %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// -----

// CHECK-LABEL: func @memref_cast_folding_subview
func @memref_cast_folding_subview(%arg0: memref<4x5xf32>, %i: index) -> (memref<?x?xf32, offset:? , strides: [?, ?]>) {
  %0 = memref_cast %arg0 : memref<4x5xf32> to memref<?x?xf32>
  // CHECK-NEXT: subview %{{.*}}: memref<4x5xf32>
  %1 = subview %0[%i, %i][%i, %i][%i, %i]: memref<?x?xf32> to memref<?x?xf32, offset:? , strides: [?, ?]>
  // CHECK-NEXT: return %{{.*}}
  return %1: memref<?x?xf32, offset:? , strides: [?, ?]>
}

// -----

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1) -> (d0 * 16 + d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

// CHECK-LABEL: func @memref_cast_folding_subview_static(
func @memref_cast_folding_subview_static(%V: memref<16x16xf32>, %a: index, %b: index)
  -> memref<3x4xf32, offset:?, strides:[?, 1]>
{
  %0 = memref_cast %V : memref<16x16xf32> to memref<?x?xf32>
  %1 = subview %0[0, 0][3, 4][1, 1] : memref<?x?xf32> to memref<3x4xf32, offset:?, strides:[?, 1]>

  // CHECK:  subview{{.*}}: memref<16x16xf32> to memref<3x4xf32, #[[$map0]]>
  // CHECK:  memref_cast{{.*}}: memref<3x4xf32, #[[$map0]]> to memref<3x4xf32, #[[$map1]]>
  return %1: memref<3x4xf32, offset:?, strides:[?, 1]>
}

// -----

// CHECK-LABEL: func @extract_element_from_tensor_from_elements
func @extract_element_from_tensor_from_elements(%element : index) -> index {
  // CHECK-SAME: ([[ARG:%.*]]: index)
  %c0 = constant 0 : index
  %tensor = tensor_from_elements %element : tensor<1xindex>
  %extracted_element = extract_element %tensor[%c0] : tensor<1xindex>
  // CHECK: [[ARG]] : index
  return %extracted_element : index
}

// -----

// CHECK-LABEL: func @extract_element_from_dynamic_tensor_from_elements
// CHECK-SAME: %[[IDX:.*]]: index, %[[TENSOR:.*]]: tensor<*xf32>
func @extract_element_from_dynamic_tensor_from_elements(%idx: index, %tensor: tensor<*xf32>) -> index {
  %size = rank %tensor : tensor<*xf32>
  // CHECK-NEXT: %[[RES:.*]] = dim %[[TENSOR]], %[[IDX]]
  %0 = dynamic_tensor_from_elements %size {
    ^bb0(%arg0: index):
    %1 = dim %tensor, %arg0 : tensor<*xf32>
    yield %1 : index
  } : tensor<?xindex>
  %1 = extract_element %0[%idx] : tensor<?xindex>
  // CHECK-NEXT: return %[[RES]]
  return %1 : index
}

// -----

// CHECK-LABEL: func @extract_element_from_dynamic_tensor_from_elements_2d
// CHECK-SAME: %[[IDX0:.*]]: index, %[[IDX1:.*]]: index, %[[TENSOR:.*]]: tensor<*xf32>
func @extract_element_from_dynamic_tensor_from_elements_2d(%idx0: index, %idx1: index, %tensor: tensor<*xf32>) -> index {
  %size = rank %tensor : tensor<*xf32>
  // CHECK-NEXT: %[[DIM0:.*]] = dim %[[TENSOR]], %[[IDX0]]
  // CHECK-NEXT: %[[DIM1:.*]] = dim %[[TENSOR]], %[[IDX1]]
  // CHECK-NEXT: %[[RES:.*]] = addi %[[DIM0]], %[[DIM1]]
  %0 = dynamic_tensor_from_elements %size, %size {
    ^bb0(%arg0: index, %arg1: index):
    %1 = dim %tensor, %arg0 : tensor<*xf32>
    %2 = dim %tensor, %arg1 : tensor<*xf32>
    %3 = addi %1, %2 : index
    yield %3 : index
  } : tensor<?x?xindex>
  %4 = extract_element %0[%idx0, %idx1] : tensor<?x?xindex>
  // CHECK-NEXT: return %[[RES]]
  return %4 : index
}

// -----

// CHECK-LABEL: func @extract_element_from_dynamic_tensor_from_elements_sideeffects
// CHECK-SAME: %[[IDX:.*]]: index
func @extract_element_from_dynamic_tensor_from_elements_sideeffects(%idx: index, %tensor: tensor<*xf32>) -> index {
  %size = rank %tensor : tensor<*xf32>
  %mem = alloc(%size) : memref<?xindex>
  // CHECK: %[[DTENSOR:.*]] = dynamic_tensor_from_elements
  %0 = dynamic_tensor_from_elements %size {
    ^bb0(%arg0: index):
    %1 = dim %tensor, %arg0 : tensor<*xf32>
    store %1, %mem[%arg0] : memref<?xindex>
    yield %1 : index
  } : tensor<?xindex>
  // CHECK: %[[RES:.*]] = extract_element %[[DTENSOR]][%[[IDX]]]
  %1 = extract_element %0[%idx] : tensor<?xindex>
  // CHECK-NEXT: return %[[RES]]
  return %1 : index
}

// -----

// CHECK-LABEL: @static_dynamic_tensor_from_elements
// CHECK-SAME: %[[SIZE1:.*]]: index, %[[SIZE4:.*]]: index)
func @static_dynamic_tensor_from_elements(%size1: index, %size4: index) -> tensor<3x?x?x7x?xindex> {
  %c5 = constant 5 : index
  // CHECK: dynamic_tensor_from_elements %[[SIZE1]], %[[SIZE4]]
  %0 = dynamic_tensor_from_elements %size1, %c5, %size4 {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
    %1 = constant 32 : index
    yield %1 : index
  // CHECK: : tensor<3x?x5x7x?xindex>
  } : tensor<3x?x?x7x?xindex>
  // CHECK: tensor_cast %{{.*}} : tensor<3x?x5x7x?xindex> to tensor<3x?x?x7x?xindex>
  return %0 : tensor<3x?x?x7x?xindex>
}

// -----

// CHECK-LABEL: @tensor_cast_chain_ok
// CHECK-SAME: %[[IN:.*]]: tensor<*xi32>
func @tensor_cast_chain_ok(%input: tensor<*xi32>) -> tensor<4x8xi32> {
  // CHECK-NEXT: %[[RES:.*]] = tensor_cast %[[IN]] : tensor<*xi32> to tensor<4x8xi32>
  %0 = tensor_cast %input : tensor<*xi32> to tensor<4x?xi32>
  %1 = tensor_cast %0 : tensor<4x?xi32> to tensor<4x8xi32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @tensor_cast_chain_regain
// CHECK-SAME: %[[IN:.*]]: tensor<4xi32>
func @tensor_cast_chain_regain(%input: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tensor_cast %input : tensor<4xi32> to tensor<?xi32>
  %1 = tensor_cast %0 : tensor<?xi32> to tensor<4xi32>
  // CHECK-NEXT: return %[[IN]]
  return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @tensor_cast_chain_keep
// CHECK-SAME: %[[IN:.*]]: tensor<?x?xi32>
func @tensor_cast_chain_keep(%input: tensor<?x?xi32>) -> tensor<?x8xi32> {
  // CHECK-NEXT: %[[C1:.*]] = tensor_cast %[[IN]]
  %0 = tensor_cast %input : tensor<?x?xi32> to tensor<4x?xi32>
  // CHECK-NEXT: %[[C2:.*]] = tensor_cast %[[C1]]
  %1 = tensor_cast %0 : tensor<4x?xi32> to tensor<?x8xi32>
  // CHECK-NEXT: return %[[C2]]
  return %1 : tensor<?x8xi32>
}

// -----

// CHECK-LABEL: @tensor_cast_chain_invalid
// CHECK-SAME: %[[IN:.*]]: tensor<4x8xi32>
func @tensor_cast_chain_invalid(%input: tensor<4x8xi32>) -> tensor<8x4xi32> {
  // CHECK-NEXT: %[[C1:.*]] = tensor_cast %[[IN]]
  %0 = tensor_cast %input : tensor<4x8xi32> to tensor<?x?xi32>
  // CHECK-NEXT: %[[C2:.*]] = tensor_cast %[[C1]]
  %1 = tensor_cast %0 : tensor<?x?xi32> to tensor<8x4xi32>
  // CHECK-NEXT: return %[[C2]]
  return %1 : tensor<8x4xi32>
}

// -----

// CHECK-LABEL: func @subtensor
// CHECK-SAME: %[[ARG0:[0-9a-z]*]]: index, %[[ARG1:[0-9a-z]*]]: index
func @subtensor(%t: tensor<8x16x4xf32>, %arg0 : index, %arg1 : index) 
  -> tensor<?x?x?xf32> 
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c7 = constant 7 : index
  %c11 = constant 11 : index

  // CHECK: subtensor %{{.*}}[0, 0, 0] [7, 11, 2] [1, 1, 1] :
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<7x11x2xf32>
  // CHECK: tensor_cast %{{.*}} : tensor<7x11x2xf32> to tensor<?x?x?xf32>
  %1 = subtensor %t[%c0, %c0, %c0] [%c7, %c11, %c2] [%c1, %c1, %c1]
    : tensor<8x16x4xf32> to tensor<?x?x?xf32>

  // Test: subtensor with one dynamic operand can also be folded.
  // CHECK: subtensor %{{.*}}[0, 0, 0] [2, %[[ARG0]], 2] [1, 1, 1] :
  // CHECK-SAME: tensor<?x?x?xf32> to tensor<2x?x2xf32>
  // CHECK: tensor_cast %{{.*}} : tensor<2x?x2xf32> to tensor<?x?x?xf32>
  %2 = subtensor %1[%c0, %c0, %c0] [%c2, %arg0, %c2] [%c1, %c1, %c1]
    : tensor<?x?x?xf32> to tensor<?x?x?xf32>

  return %2 : tensor<?x?x?xf32>
}
