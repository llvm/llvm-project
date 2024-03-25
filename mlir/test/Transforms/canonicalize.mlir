// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_subi_zero
func.func @test_subi_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT: return %c0
  %y = arith.subi %arg0, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: func @test_subi_zero_vector
func.func @test_subi_zero_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  //CHECK-NEXT: %cst = arith.constant dense<0> : vector<4xi32>
  %y = arith.subi %arg0, %arg0 : vector<4xi32>
  // CHECK-NEXT: return %cst
  return %y: vector<4xi32>
}

// CHECK-LABEL: func @test_subi_zero_tensor
func.func @test_subi_zero_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  //CHECK-NEXT: %cst = arith.constant dense<0> : tensor<4x5xi32>
  %y = arith.subi %arg0, %arg0 : tensor<4x5xi32>
  // CHECK-NEXT: return %cst
  return %y: tensor<4x5xi32>
}

// CHECK-LABEL: func @dim
func.func @dim(%arg0: tensor<8x4xf32>) -> index {

  // CHECK: %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c1 : tensor<8x4xf32>

  // CHECK-NEXT: return %c4
  return %0 : index
}

// CHECK-LABEL: func @test_commutative
func.func @test_commutative(%arg0: i32) -> (i32, i32) {
  // CHECK: %c42_i32 = arith.constant 42 : i32
  %c42_i32 = arith.constant 42 : i32
  // CHECK-NEXT: %0 = arith.addi %arg0, %c42_i32 : i32
  %y = arith.addi %c42_i32, %arg0 : i32

  // This should not be swapped.
  // CHECK-NEXT: %1 = arith.subi %c42_i32, %arg0 : i32
  %z = arith.subi %c42_i32, %arg0 : i32

  // CHECK-NEXT: return %0, %1
  return %y, %z: i32, i32
}

// CHECK-LABEL: func @trivial_dce
func.func @trivial_dce(%arg0: tensor<8x4xf32>) {
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c1 : tensor<8x4xf32>
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @load_dce
func.func @load_dce(%arg0: index) {
  %c4 = arith.constant 4 : index
  %a = memref.alloc(%c4) : memref<?xf32>
  %2 = memref.load %a[%arg0] : memref<?xf32>
  memref.dealloc %a: memref<?xf32>
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @addi_zero
func.func @addi_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: return %arg0
  %c0_i32 = arith.constant 0 : i32
  %y = arith.addi %c0_i32, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: func @addi_zero_index
func.func @addi_zero_index(%arg0: index) -> index {
  // CHECK-NEXT: return %arg0
  %c0_index = arith.constant 0 : index
  %y = arith.addi %c0_index, %arg0 : index
  return %y: index
}


// CHECK-LABEL: func @addi_zero_vector
func.func @addi_zero_vector(%arg0: vector<4 x i32>) -> vector<4 x i32> {
  // CHECK-NEXT: return %arg0
  %c0_v4i32 = arith.constant dense<0> : vector<4 x i32>
  %y = arith.addi %c0_v4i32, %arg0 : vector<4 x i32>
  return %y: vector<4 x i32>
}

// CHECK-LABEL: func @addi_zero_tensor
func.func @addi_zero_tensor(%arg0: tensor<4 x 5 x i32>) -> tensor<4 x 5 x i32> {
  // CHECK-NEXT: return %arg0
  %c0_t45i32 = arith.constant dense<0> : tensor<4 x 5 x i32>
  %y = arith.addi %arg0, %c0_t45i32 : tensor<4 x 5 x i32>
  return %y: tensor<4 x 5 x i32>
}

// CHECK-LABEL: func @muli_zero
func.func @muli_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32

  %y = arith.muli %c0_i32, %arg0 : i32

  // CHECK-NEXT: return %c0_i32
  return %y: i32
}

// CHECK-LABEL: func @muli_zero_index
func.func @muli_zero_index(%arg0: index) -> index {
  // CHECK-NEXT: %[[CST:.*]] = arith.constant 0 : index
  %c0_index = arith.constant 0 : index

  %y = arith.muli %c0_index, %arg0 : index

  // CHECK-NEXT: return %[[CST]]
  return %y: index
}

// CHECK-LABEL: func @muli_zero_vector
func.func @muli_zero_vector(%arg0: vector<4 x i32>) -> vector<4 x i32> {
  // CHECK-NEXT: %cst = arith.constant dense<0> : vector<4xi32>
  %cst = arith.constant dense<0> : vector<4 x i32>

  %y = arith.muli %cst, %arg0 : vector<4 x i32>

  // CHECK-NEXT: return %cst
  return %y: vector<4 x i32>
}

// CHECK-LABEL: func @muli_zero_tensor
func.func @muli_zero_tensor(%arg0: tensor<4 x 5 x i32>) -> tensor<4 x 5 x i32> {
  // CHECK-NEXT: %cst = arith.constant dense<0> : tensor<4x5xi32>
  %cst = arith.constant dense<0> : tensor<4 x 5 x i32>

  %y = arith.muli %arg0, %cst : tensor<4 x 5 x i32>

  // CHECK-NEXT: return %cst
  return %y: tensor<4 x 5 x i32>
}

// CHECK-LABEL: func @muli_one
func.func @muli_one(%arg0: i32) -> i32 {
  // CHECK-NEXT: return %arg0
  %c0_i32 = arith.constant 1 : i32
  %y = arith.muli %c0_i32, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: func @muli_one_index
func.func @muli_one_index(%arg0: index) -> index {
  // CHECK-NEXT: return %arg0
  %c0_index = arith.constant 1 : index
  %y = arith.muli %c0_index, %arg0 : index
  return %y: index
}

// CHECK-LABEL: func @muli_one_vector
func.func @muli_one_vector(%arg0: vector<4 x i32>) -> vector<4 x i32> {
  // CHECK-NEXT: return %arg0
  %c1_v4i32 = arith.constant dense<1> : vector<4 x i32>
  %y = arith.muli %c1_v4i32, %arg0 : vector<4 x i32>
  return %y: vector<4 x i32>
}

// CHECK-LABEL: func @muli_one_tensor
func.func @muli_one_tensor(%arg0: tensor<4 x 5 x i32>) -> tensor<4 x 5 x i32> {
  // CHECK-NEXT: return %arg0
  %c1_t45i32 = arith.constant dense<1> : tensor<4 x 5 x i32>
  %y = arith.muli %arg0, %c1_t45i32 : tensor<4 x 5 x i32>
  return %y: tensor<4 x 5 x i32>
}

//CHECK-LABEL: func @and_self
func.func @and_self(%arg0: i32) -> i32 {
  //CHECK-NEXT: return %arg0
  %1 = arith.andi %arg0, %arg0 : i32
  return %1 : i32
}

//CHECK-LABEL: func @and_self_vector
func.func @and_self_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  //CHECK-NEXT: return %arg0
  %1 = arith.andi %arg0, %arg0 : vector<4xi32>
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @and_self_tensor
func.func @and_self_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  //CHECK-NEXT: return %arg0
  %1 = arith.andi %arg0, %arg0 : tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

//CHECK-LABEL: func @and_zero
func.func @and_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT: return %c0_i32
  %1 = arith.andi %arg0, %c0_i32 : i32
  return %1 : i32
}

//CHECK-LABEL: func @and_zero_index
func.func @and_zero_index(%arg0: index) -> index {
  // CHECK-NEXT: %[[CST:.*]] = arith.constant 0 : index
  %c0_index = arith.constant 0 : index
  // CHECK-NEXT: return %[[CST]]
  %1 = arith.andi %arg0, %c0_index : index
  return %1 : index
}

//CHECK-LABEL: func @and_zero_vector
func.func @and_zero_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  // CHECK-NEXT: %cst = arith.constant dense<0> : vector<4xi32>
  %cst = arith.constant dense<0> : vector<4xi32>
  // CHECK-NEXT: return %cst
  %1 = arith.andi %arg0, %cst : vector<4xi32>
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @and_zero_tensor
func.func @and_zero_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  // CHECK-NEXT: %cst = arith.constant dense<0> : tensor<4x5xi32>
  %cst = arith.constant dense<0> : tensor<4x5xi32>
  // CHECK-NEXT: return %cst
  %1 = arith.andi %arg0, %cst : tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

//CHECK-LABEL: func @or_self
func.func @or_self(%arg0: i32) -> i32 {
  //CHECK-NEXT: return %arg0
  %1 = arith.ori %arg0, %arg0 : i32
  return %1 : i32
}

//CHECK-LABEL: func @or_self_vector
func.func @or_self_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  //CHECK-NEXT: return %arg0
  %1 = arith.ori %arg0, %arg0 : vector<4xi32>
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @or_self_tensor
func.func @or_self_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  //CHECK-NEXT: return %arg0
  %1 = arith.ori %arg0, %arg0 : tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

//CHECK-LABEL: func @or_zero
func.func @or_zero(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT: return %arg0
  %1 = arith.ori %arg0, %c0_i32 : i32
  return %1 : i32
}

//CHECK-LABEL: func @or_zero_index
func.func @or_zero_index(%arg0: index) -> index {
  %c0_index = arith.constant 0 : index
  // CHECK-NEXT: return %arg0
  %1 = arith.ori %arg0, %c0_index : index
  return %1 : index
}

//CHECK-LABEL: func @or_zero_vector
func.func @or_zero_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  // CHECK-NEXT: return %arg0
  %cst = arith.constant dense<0> : vector<4xi32>
  %1 = arith.ori %arg0, %cst : vector<4xi32>
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @or_zero_tensor
func.func @or_zero_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  // CHECK-NEXT: return %arg0
  %cst = arith.constant dense<0> : tensor<4x5xi32>
  %1 = arith.ori %arg0, %cst : tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

// CHECK-LABEL: func @or_all_ones
func.func @or_all_ones(%arg0: i1, %arg1: i4) -> (i1, i4) {
  // CHECK-DAG: %c-1_i4 = arith.constant -1 : i4
  // CHECK-DAG: %true = arith.constant true
  %c1_i1 = arith.constant 1 : i1
  %c15 = arith.constant 15 : i4
  // CHECK-NEXT: return %true
  %1 = arith.ori %arg0, %c1_i1 : i1
  %2 = arith.ori %arg1, %c15 : i4
  return %1, %2 : i1, i4
}

//CHECK-LABEL: func @xor_self
func.func @xor_self(%arg0: i32) -> i32 {
  //CHECK-NEXT: %c0_i32 = arith.constant 0
  %1 = arith.xori %arg0, %arg0 : i32
  //CHECK-NEXT: return %c0_i32
  return %1 : i32
}

//CHECK-LABEL: func @xor_self_vector
func.func @xor_self_vector(%arg0: vector<4xi32>) -> vector<4xi32> {
  //CHECK-NEXT: %cst = arith.constant dense<0> : vector<4xi32>
  %1 = arith.xori %arg0, %arg0 : vector<4xi32>
  //CHECK-NEXT: return %cst
  return %1 : vector<4xi32>
}

//CHECK-LABEL: func @xor_self_tensor
func.func @xor_self_tensor(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  //CHECK-NEXT: %cst = arith.constant dense<0> : tensor<4x5xi32>
  %1 = arith.xori %arg0, %arg0 : tensor<4x5xi32>
  //CHECK-NEXT: return %cst
  return %1 : tensor<4x5xi32>
}

// CHECK-LABEL: func @memref_cast_folding
func.func @memref_cast_folding(%arg0: memref<4 x f32>, %arg1: f32) -> (f32, f32) {
  %0 = memref.cast %arg0 : memref<4xf32> to memref<?xf32>
  // CHECK-NEXT: %c0 = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dim = memref.dim %0, %c0 : memref<? x f32>

  // CHECK-NEXT: affine.load %arg0[3]
  %1 = affine.load %0[%dim - 1] : memref<?xf32>

  // CHECK-NEXT: memref.store %arg1, %arg0[%c0] : memref<4xf32>
  memref.store %arg1, %0[%c0] : memref<?xf32>

  // CHECK-NEXT: %{{.*}} = memref.load %arg0[%c0] : memref<4xf32>
  %2 = memref.load %0[%c0] : memref<?xf32>

  // CHECK-NEXT: memref.dealloc %arg0 : memref<4xf32>
  memref.dealloc %0: memref<?xf32>

  // CHECK-NEXT: return %{{.*}}
  return %1, %2 : f32, f32
}

// CHECK-LABEL: @fold_memref_cast_in_memref_cast
// CHECK-SAME: (%[[ARG0:.*]]: memref<42x42xf64>)
func.func @fold_memref_cast_in_memref_cast(%0: memref<42x42xf64>) {
  // CHECK: %[[folded:.*]] = memref.cast %[[ARG0]] : memref<42x42xf64> to memref<?x?xf64>
  %4 = memref.cast %0 : memref<42x42xf64> to memref<?x42xf64>
  // CHECK-NOT: memref.cast
  %5 = memref.cast %4 : memref<?x42xf64> to memref<?x?xf64>
  // CHECK: "test.user"(%[[folded]])
  "test.user"(%5) : (memref<?x?xf64>) -> ()
  return
}

// CHECK-LABEL: @fold_memref_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: memref<42x42xf64>)
func.func @fold_memref_cast_chain(%0: memref<42x42xf64>) {
  // CHECK-NOT: memref.cast
  %4 = memref.cast %0 : memref<42x42xf64> to memref<?x42xf64>
  %5 = memref.cast %4 : memref<?x42xf64> to memref<42x42xf64>
  // CHECK: "test.user"(%[[ARG0]])
  "test.user"(%5) : (memref<42x42xf64>) -> ()
  return
}

// CHECK-LABEL: func @dead_alloc_fold
func.func @dead_alloc_fold() {
  // CHECK-NEXT: return
  %c4 = arith.constant 4 : index
  %a = memref.alloc(%c4) : memref<?xf32>
  return
}

// CHECK-LABEL: func @dead_dealloc_fold
func.func @dead_dealloc_fold() {
  // CHECK-NEXT: return
  %a = memref.alloc() : memref<4xf32>
  memref.dealloc %a: memref<4xf32>
  return
}

// CHECK-LABEL: func @dead_dealloc_fold_multi_use
func.func @dead_dealloc_fold_multi_use(%cond : i1) {
  // CHECK-NEXT: return
  %a = memref.alloc() : memref<4xf32>
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  memref.dealloc %a: memref<4xf32>
  return

^bb2:
  memref.dealloc %a: memref<4xf32>
  return
}

// CHECK-LABEL: func @write_only_alloc_fold
func.func @write_only_alloc_fold(%v: f32) {
  // CHECK-NEXT: return
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %a = memref.alloc(%c4) : memref<?xf32>
  memref.store %v, %a[%c0] : memref<?xf32>
  memref.dealloc %a: memref<?xf32>
  return
}

// CHECK-LABEL: func @write_only_alloca_fold
func.func @write_only_alloca_fold(%v: f32) {
  // CHECK-NEXT: return
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %a = memref.alloca(%c4) : memref<?xf32>
  memref.store %v, %a[%c0] : memref<?xf32>
  return
}

// CHECK-LABEL: func @dead_block_elim
func.func @dead_block_elim() {
  // CHECK-NOT: ^bb
  builtin.module {
    func.func @nested() {
      return

    ^bb1:
      return
    }
  }
  return
}

// CHECK-LABEL: func @dyn_shape_fold(%arg0: index, %arg1: index)
func.func @dyn_shape_fold(%L : index, %M : index) -> (memref<4 x ? x 8 x ? x ? x f32>, memref<? x ? x i32>, memref<? x ? x f32>, memref<4 x ? x 8 x ? x ? x f32>) {
  // CHECK: %c0 = arith.constant 0 : index
  %zero = arith.constant 0 : index
  // The constants below disappear after they propagate into shapes.
  %nine = arith.constant 9 : index
  %N = arith.constant 1024 : index
  %K = arith.constant 512 : index

  // CHECK: memref.alloc(%arg0) : memref<?x1024xf32>
  %a = memref.alloc(%L, %N) : memref<? x ? x f32>

  // CHECK: memref.alloc(%arg1) : memref<4x1024x8x512x?xf32>
  %b = memref.alloc(%N, %K, %M) : memref<4 x ? x 8 x ? x ? x f32>

  // CHECK: memref.alloc() : memref<512x1024xi32>
  %c = memref.alloc(%K, %N) : memref<? x ? x i32>

  // CHECK: memref.alloc() : memref<9x9xf32>
  %d = memref.alloc(%nine, %nine) : memref<? x ? x f32>

  // CHECK: memref.alloca(%arg1) : memref<4x1024x8x512x?xf32>
  %e = memref.alloca(%N, %K, %M) : memref<4 x ? x 8 x ? x ? x f32>

  // CHECK: affine.for
  affine.for %i = 0 to %L {
    // CHECK-NEXT: affine.for
    affine.for %j = 0 to 10 {
      // CHECK-NEXT: memref.load %{{.*}}[%arg2, %arg3] : memref<?x1024xf32>
      // CHECK-NEXT: memref.store %{{.*}}, %{{.*}}[%c0, %c0, %arg2, %arg3, %c0] : memref<4x1024x8x512x?xf32>
      %v = memref.load %a[%i, %j] : memref<?x?xf32>
      memref.store %v, %b[%zero, %zero, %i, %j, %zero] : memref<4x?x8x?x?xf32>
    }
  }

  return %b, %c, %d, %e : memref<4 x ? x 8 x ? x ? x f32>, memref<? x ? x i32>, memref<? x ? x f32>, memref<4 x ? x 8 x ? x ? x f32>
}

// CHECK-LABEL: func @dim_op_fold(
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: index
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: index
// CHECK-SAME: %[[ARG2:[a-z0-9]*]]: index
// CHECK-SAME: %[[BUF:[a-z0-9]*]]: memref<?xi8>
func.func @dim_op_fold(%arg0: index, %arg1: index, %arg2: index, %BUF: memref<?xi8>, %M : index, %N : index, %K : index) {
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %1 = memref.alloc(%arg1, %arg2) : memref<?x8x?xf32>
  %2 = memref.dim %1, %c2 : memref<?x8x?xf32>
  affine.for %arg3 = 0 to %2 {
    %3 = memref.alloc(%arg0) : memref<?xi8>
    %ub = memref.dim %3, %c0 : memref<?xi8>
    affine.for %arg4 = 0 to %ub {
      %s = memref.dim %0, %c0 : memref<?x?xf32>
      %v = memref.view %3[%c0][%arg4, %s] : memref<?xi8> to memref<?x?xf32>
      %sv = memref.subview %0[%c0, %c0][%s,%arg4][%c1,%c1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %l = memref.dim %v, %c1 : memref<?x?xf32>
      %u = memref.dim %sv, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
      affine.for %arg5 = %l to %u {
        "foo"() : () -> ()
      }
      %sv2 = memref.subview %0[0, 0][17, %arg4][1, 1] : memref<?x?xf32> to memref<17x?xf32, strided<[?, 1], offset: ?>>
      %l2 = memref.dim %v, %c1 : memref<?x?xf32>
      %u2 = memref.dim %sv2, %c1 : memref<17x?xf32, strided<[?, 1], offset: ?>>
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

  %A = memref.view %BUF[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32>
  %B = memref.view %BUF[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32>
  %C = memref.view %BUF[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>

  %M_ = memref.dim %A, %c0 : memref<?x?xf32>
  %K_ = memref.dim %A, %c1 : memref<?x?xf32>
  %N_ = memref.dim %C, %c1 : memref<?x?xf32>
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
func.func @merge_constants() -> (index, index) {
  // CHECK-NEXT: %c42 = arith.constant 42 : index
  %0 = arith.constant 42 : index
  %1 = arith.constant 42 : index
  // CHECK-NEXT: return %c42, %c42
  return %0, %1: index, index
}

// CHECK-LABEL: func @hoist_constant
func.func @hoist_constant(%arg0: memref<8xi32>) {
  // CHECK-NEXT: %c42_i32 = arith.constant 42 : i32
  // CHECK-NEXT: affine.for %arg1 = 0 to 8 {
  affine.for %arg1 = 0 to 8 {
    // CHECK-NEXT: memref.store %c42_i32, %arg0[%arg1]
    %c42_i32 = arith.constant 42 : i32
    memref.store %c42_i32, %arg0[%arg1] : memref<8xi32>
  }
  return
}

// CHECK-LABEL: func @const_fold_propagate
func.func @const_fold_propagate() -> memref<?x?xf32> {
  %VT_i = arith.constant 512 : index

  %VT_i_s = affine.apply affine_map<(d0) -> (d0 floordiv  8)> (%VT_i)
  %VT_k_l = affine.apply affine_map<(d0) -> (d0 floordiv  16)> (%VT_i)

  // CHECK: = memref.alloc() : memref<64x32xf32>
  %Av = memref.alloc(%VT_i_s, %VT_k_l) : memref<?x?xf32>
  return %Av : memref<?x?xf32>
}

// CHECK-LABEL: func @indirect_call_folding
func.func @indirect_target() {
  return
}

func.func @indirect_call_folding() {
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
func.func @lowered_affine_mod() -> (index, index) {
// CHECK-DAG: {{.*}} = arith.constant 1 : index
// CHECK-DAG: {{.*}} = arith.constant 41 : index
  %c-43 = arith.constant -43 : index
  %c42 = arith.constant 42 : index
  %0 = arith.remsi %c-43, %c42 : index
  %c0 = arith.constant 0 : index
  %1 = arith.cmpi slt, %0, %c0 : index
  %2 = arith.addi %0, %c42 : index
  %3 = arith.select %1, %2, %0 : index
  %c43 = arith.constant 43 : index
  %c42_0 = arith.constant 42 : index
  %4 = arith.remsi %c43, %c42_0 : index
  %c0_1 = arith.constant 0 : index
  %5 = arith.cmpi slt, %4, %c0_1 : index
  %6 = arith.addi %4, %c42_0 : index
  %7 = arith.select %5, %6, %4 : index
  return %3, %7 : index, index
}

//
// IMPORTANT NOTE: the operations in this test are exactly those produced by
// lowering affine.apply affine_map<(i) -> (i mod 42)> to standard operations.  Please only
// change these operations together with the affine lowering pass tests.
//
// CHECK-LABEL: func @lowered_affine_floordiv
func.func @lowered_affine_floordiv() -> (index, index) {
// CHECK-DAG: %c1 = arith.constant 1 : index
// CHECK-DAG: %c-2 = arith.constant -2 : index
  %c-43 = arith.constant -43 : index
  %c42 = arith.constant 42 : index
  %c0 = arith.constant 0 : index
  %c-1 = arith.constant -1 : index
  %0 = arith.cmpi slt, %c-43, %c0 : index
  %1 = arith.subi %c-1, %c-43 : index
  %2 = arith.select %0, %1, %c-43 : index
  %3 = arith.divsi %2, %c42 : index
  %4 = arith.subi %c-1, %3 : index
  %5 = arith.select %0, %4, %3 : index
  %c43 = arith.constant 43 : index
  %c42_0 = arith.constant 42 : index
  %c0_1 = arith.constant 0 : index
  %c-1_2 = arith.constant -1 : index
  %6 = arith.cmpi slt, %c43, %c0_1 : index
  %7 = arith.subi %c-1_2, %c43 : index
  %8 = arith.select %6, %7, %c43 : index
  %9 = arith.divsi %8, %c42_0 : index
  %10 = arith.subi %c-1_2, %9 : index
  %11 = arith.select %6, %10, %9 : index
  return %5, %11 : index, index
}

//
// IMPORTANT NOTE: the operations in this test are exactly those produced by
// lowering affine.apply affine_map<(i) -> (i mod 42)> to standard operations.  Please only
// change these operations together with the affine lowering pass tests.
//
// CHECK-LABEL: func @lowered_affine_ceildiv
func.func @lowered_affine_ceildiv() -> (index, index) {
// CHECK-DAG:  %c-1 = arith.constant -1 : index
  %c-43 = arith.constant -43 : index
  %c42 = arith.constant 42 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.cmpi sle, %c-43, %c0 : index
  %1 = arith.subi %c0, %c-43 : index
  %2 = arith.subi %c-43, %c1 : index
  %3 = arith.select %0, %1, %2 : index
  %4 = arith.divsi %3, %c42 : index
  %5 = arith.subi %c0, %4 : index
  %6 = arith.addi %4, %c1 : index
  %7 = arith.select %0, %5, %6 : index
// CHECK-DAG:  %c2 = arith.constant 2 : index
  %c43 = arith.constant 43 : index
  %c42_0 = arith.constant 42 : index
  %c0_1 = arith.constant 0 : index
  %c1_2 = arith.constant 1 : index
  %8 = arith.cmpi sle, %c43, %c0_1 : index
  %9 = arith.subi %c0_1, %c43 : index
  %10 = arith.subi %c43, %c1_2 : index
  %11 = arith.select %8, %9, %10 : index
  %12 = arith.divsi %11, %c42_0 : index
  %13 = arith.subi %c0_1, %12 : index
  %14 = arith.addi %12, %c1_2 : index
  %15 = arith.select %8, %13, %14 : index

  // CHECK-NEXT: return %c-1, %c2
  return %7, %15 : index, index
}

// Checks that NOP casts are removed.
// CHECK-LABEL: cast_values
func.func @cast_values(%arg0: memref<?xi32>) -> memref<2xi32> {
  // NOP cast
  %1 = memref.cast %arg0 : memref<?xi32> to memref<?xi32>
  // CHECK-NEXT: %[[RET:.*]] = memref.cast %arg0 : memref<?xi32> to memref<2xi32>
  %3 = memref.cast %1 : memref<?xi32> to memref<2xi32>
  // NOP cast
  %5 = memref.cast %3 : memref<2xi32> to memref<2xi32>
  // CHECK-NEXT: return %[[RET]] : memref<2xi32>
  return %5 : memref<2xi32>
}

// -----

// CHECK-LABEL: func @view
func.func @view(%arg0 : index) -> (f32, f32, f32, f32) {
  // CHECK: %[[C15:.*]] = arith.constant 15 : index
  // CHECK: %[[ALLOC_MEM:.*]] = memref.alloc() : memref<2048xi8>
  %0 = memref.alloc() : memref<2048xi8>
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index
  %c15 = arith.constant 15 : index

  // Test: fold constant sizes.
  // CHECK: memref.view %[[ALLOC_MEM]][%[[C15]]][] : memref<2048xi8> to memref<7x11xf32>
  %1 = memref.view %0[%c15][%c7, %c11] : memref<2048xi8> to memref<?x?xf32>
  %r0 = memref.load %1[%c0, %c0] : memref<?x?xf32>

  // Test: fold one constant size.
  // CHECK: memref.view %[[ALLOC_MEM]][%[[C15]]][%arg0, %arg0] : memref<2048xi8> to memref<?x?x7xf32>
  %2 = memref.view %0[%c15][%arg0, %arg0, %c7] : memref<2048xi8> to memref<?x?x?xf32>
  %r1 = memref.load %2[%c0, %c0, %c0] : memref<?x?x?xf32>

  // Test: preserve an existing static size.
  // CHECK: memref.view %[[ALLOC_MEM]][%[[C15]]][] : memref<2048xi8> to memref<7x4xf32>
  %3 = memref.view %0[%c15][%c7] : memref<2048xi8> to memref<?x4xf32>
  %r2 = memref.load %3[%c0, %c0] : memref<?x4xf32>

  // Test: folding static alloc and memref.cast into a view.
  // CHECK memref.view %[[ALLOC_MEM]][%[[C15]]][] : memref<2048xi8> to memref<15x7xf32>
  %4 = memref.cast %0 : memref<2048xi8> to memref<?xi8>
  %5 = memref.view %4[%c15][%c15, %c7] : memref<?xi8> to memref<?x?xf32>
  %r3 = memref.load %5[%c0, %c0] : memref<?x?xf32>
  return %r0, %r1, %r2, %r3 : f32, f32, f32, f32
}

// -----

// CHECK-LABEL: func @subview
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @subview(%arg0 : index, %arg1 : index) -> (index, index) {
  // Folded but reappears after subview folding into dim.
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C7:.*]] = arith.constant 7 : index
  // CHECK-DAG: %[[C11:.*]] = arith.constant 11 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT: arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: arith.constant 2 : index
  %c2 = arith.constant 2 : index
  // Folded but reappears after subview folding into dim.
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index
  // CHECK-NOT: arith.constant 15 : index
  %c15 = arith.constant 15 : index

  // CHECK: %[[ALLOC0:.*]] = memref.alloc()
  %0 = memref.alloc() : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>>

  // Test: subview with constant base memref and constant operands is folded.
  // Note that the subview uses the base memrefs layout map because it used
  // zero offset and unit stride arguments.
  // CHECK: memref.subview %[[ALLOC0]][0, 0, 0] [7, 11, 2] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>>
  // CHECK-SAME: to memref<7x11x2xf32, strided<[64, 4, 1]>>
  %1 = memref.subview %0[%c0, %c0, %c0] [%c7, %c11, %c2] [%c1, %c1, %c1]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %v0 = memref.load %1[%c0, %c0, %c0] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test: subview with one dynamic operand can also be folded.
  // CHECK: memref.subview %[[ALLOC0]][0, %[[ARG0]], 0] [7, 11, 15] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>>
  // CHECK-SAME: to memref<7x11x15xf32, strided<[64, 4, 1], offset: ?>>
  %2 = memref.subview %0[%c0, %arg0, %c0] [%c7, %c11, %c15] [%c1, %c1, %c1]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %2[%c0, %c0, %c0] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // CHECK: %[[ALLOC1:.*]] = memref.alloc(%[[ARG0]])
  %3 = memref.alloc(%arg0) : memref<?x16x4xf32, strided<[64, 4, 1], offset: 0>>
  // Test: subview with constant operands but dynamic base memref is folded as long as the strides and offset of the base memref are static.
  // CHECK: memref.subview %[[ALLOC1]][0, 0, 0] [7, 11, 15] [1, 1, 1] :
  // CHECK-SAME: memref<?x16x4xf32, strided<[64, 4, 1]>>
  // CHECK-SAME: to memref<7x11x15xf32, strided<[64, 4, 1]>>
  %4 = memref.subview %3[%c0, %c0, %c0] [%c7, %c11, %c15] [%c1, %c1, %c1]
    : memref<?x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %4[%c0, %c0, %c0] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test: subview offset operands are folded correctly w.r.t. base strides.
  // CHECK: memref.subview %[[ALLOC0]][1, 2, 7] [7, 11, 2] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>> to
  // CHECK-SAME: memref<7x11x2xf32, strided<[64, 4, 1], offset: 79>>
  %5 = memref.subview %0[%c1, %c2, %c7] [%c7, %c11, %c2] [%c1, %c1, %c1]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %5[%c0, %c0, %c0] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test: subview stride operands are folded correctly w.r.t. base strides.
  // CHECK: memref.subview %[[ALLOC0]][0, 0, 0] [7, 11, 2] [2, 7, 11] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>>
  // CHECK-SAME: to memref<7x11x2xf32, strided<[128, 28, 11]>>
  %6 = memref.subview %0[%c0, %c0, %c0] [%c7, %c11, %c2] [%c2, %c7, %c11]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %6[%c0, %c0, %c0] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test: subview shape are folded, but offsets and strides are not even if base memref is static
  // CHECK: memref.subview %[[ALLOC0]][%[[ARG0]], %[[ARG0]], %[[ARG0]]] [7, 11, 2] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>> to
  // CHECK-SAME: memref<7x11x2xf32, strided<[?, ?, ?], offset: ?>>
  %10 = memref.subview %0[%arg0, %arg0, %arg0] [%c7, %c11, %c2] [%arg1, %arg1, %arg1] :
    memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %10[%arg1, %arg1, %arg1] :
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test: subview strides are folded, but offsets and shape are not even if base memref is static
  // CHECK: memref.subview %[[ALLOC0]][%[[ARG0]], %[[ARG0]], %[[ARG0]]] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] [2, 7, 11] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>> to
  // CHECK-SAME: memref<?x?x?xf32, strided<[128, 28, 11], offset: ?>>
  %11 = memref.subview %0[%arg0, %arg0, %arg0] [%arg1, %arg1, %arg1] [%c2, %c7, %c11] :
    memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %11[%arg0, %arg0, %arg0] :
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test: subview offsets are folded, but strides and shape are not even if base memref is static
  // CHECK: memref.subview %[[ALLOC0]][1, 2, 7] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] [%[[ARG0]], %[[ARG0]], %[[ARG0]]] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>> to
  // CHECK-SAME: memref<?x?x?xf32, strided<[?, ?, ?], offset: 79>>
  %13 = memref.subview %0[%c1, %c2, %c7] [%arg1, %arg1, %arg1] [%arg0, %arg0, %arg0] :
    memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %13[%arg1, %arg1, %arg1] :
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // CHECK: %[[ALLOC2:.*]] = memref.alloc(%[[ARG0]], %[[ARG0]], %[[ARG1]])
  %14 = memref.alloc(%arg0, %arg0, %arg1) : memref<?x?x?xf32>
  // Test: subview shape are folded, even if base memref is not static
  // CHECK: memref.subview %[[ALLOC2]][%[[ARG0]], %[[ARG0]], %[[ARG0]]] [7, 11, 2] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] :
  // CHECK-SAME: memref<?x?x?xf32> to
  // CHECK-SAME: memref<7x11x2xf32, strided<[?, ?, ?], offset: ?>>
  %15 = memref.subview %14[%arg0, %arg0, %arg0] [%c7, %c11, %c2] [%arg1, %arg1, %arg1] :
    memref<?x?x?xf32> to
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %15[%arg1, %arg1, %arg1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // TEST: subview strides are folded, in the type only the most minor stride is folded.
  // CHECK: memref.subview %[[ALLOC2]][%[[ARG0]], %[[ARG0]], %[[ARG0]]] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] [2, 2, 2] :
  // CHECK-SAME: memref<?x?x?xf32> to
  // CHECK-SAME: memref<?x?x?xf32, strided<[?, ?, 2], offset: ?>>
  %16 = memref.subview %14[%arg0, %arg0, %arg0] [%arg1, %arg1, %arg1] [%c2, %c2, %c2] :
    memref<?x?x?xf32> to
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %16[%arg0, %arg0, %arg0] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // TEST: subview offsets are folded but the type offset remains dynamic, when the base memref is not static
  // CHECK: memref.subview %[[ALLOC2]][1, 1, 1] [%[[ARG0]], %[[ARG0]], %[[ARG0]]] [%[[ARG1]], %[[ARG1]], %[[ARG1]]] :
  // CHECK-SAME: memref<?x?x?xf32> to
  // CHECK-SAME: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %17 = memref.subview %14[%c1, %c1, %c1] [%arg0, %arg0, %arg0] [%arg1, %arg1, %arg1] :
    memref<?x?x?xf32> to
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  memref.store %v0, %17[%arg0, %arg0, %arg0] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // CHECK: %[[ALLOC3:.*]] = memref.alloc() : memref<12x4xf32>
  %18 = memref.alloc() : memref<12x4xf32>
  %c4 = arith.constant 4 : index

  // TEST: subview strides are maintained when sizes are folded
  // CHECK: memref.subview %[[ALLOC3]][%arg1, %arg1] [2, 4] [1, 1] :
  // CHECK-SAME: memref<12x4xf32> to
  // CHECK-SAME: memref<2x4xf32, strided<[4, 1], offset: ?>
  %19 = memref.subview %18[%arg1, %arg1] [%c2, %c4] [1, 1] :
    memref<12x4xf32> to
    memref<?x?xf32, strided<[4, 1], offset: ?>>
  memref.store %v0, %19[%arg1, %arg1] : memref<?x?xf32, strided<[4, 1], offset: ?>>

  // TEST: subview strides and sizes are maintained when offsets are folded
  // CHECK: memref.subview %[[ALLOC3]][2, 4] [12, 4] [1, 1] :
  // CHECK-SAME: memref<12x4xf32> to
  // CHECK-SAME: memref<12x4xf32, strided<[4, 1], offset: 12>>
  %20 = memref.subview %18[%c2, %c4] [12, 4] [1, 1] :
    memref<12x4xf32> to
    memref<12x4xf32, strided<[4, 1], offset: ?>>
  memref.store %v0, %20[%arg1, %arg1] : memref<12x4xf32, strided<[4, 1], offset: ?>>

  // Test: dim on subview is rewritten to size operand.
  %7 = memref.dim %4, %c0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %8 = memref.dim %4, %c1 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // CHECK: return %[[C7]], %[[C11]]
  return %7, %8 : index, index
}

// CHECK-LABEL: func @index_cast
// CHECK-SAME: %[[ARG_0:arg[0-9]+]]: i16
func.func @index_cast(%arg0: i16) -> (i16) {
  %11 = arith.index_cast %arg0 : i16 to index
  %12 = arith.index_cast %11 : index to i16
  // CHECK: return %[[ARG_0]] : i16
  return %12 : i16
}

// CHECK-LABEL: func @index_cast_fold
func.func @index_cast_fold() -> (i16, index) {
  %c4 = arith.constant 4 : index
  %1 = arith.index_cast %c4 : index to i16
  %c4_i16 = arith.constant 4 : i16
  %2 = arith.index_cast %c4_i16 : i16 to index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[C4_I16:.*]] = arith.constant 4 : i16
  // CHECK: return %[[C4_I16]], %[[C4]] : i16, index
  return %1, %2 : i16, index
}

// CHECK-LABEL: func @remove_dead_else
func.func @remove_dead_else(%M : memref<100 x i32>) {
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
func.func @divi_signed_by_one(%arg0: i32) -> (i32) {
  %c1 = arith.constant 1 : i32
  %res = arith.divsi %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @divi_unsigned_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @divi_unsigned_by_one(%arg0: i32) -> (i32) {
  %c1 = arith.constant 1 : i32
  %res = arith.divui %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @tensor_divi_signed_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @tensor_divi_signed_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = arith.constant dense<1> : tensor<4x5xi32>
  %res = arith.divsi %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// CHECK-LABEL: func @tensor_divi_unsigned_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @tensor_divi_unsigned_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = arith.constant dense<1> : tensor<4x5xi32>
  %res = arith.divui %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// -----

// CHECK-LABEL: func @arith.floordivsi_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @arith.floordivsi_by_one(%arg0: i32) -> (i32) {
  %c1 = arith.constant 1 : i32
  %res = arith.floordivsi %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @tensor_arith.floordivsi_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @tensor_arith.floordivsi_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = arith.constant dense<1> : tensor<4x5xi32>
  %res = arith.floordivsi %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// CHECK-LABEL: func @arith.floordivsi_by_one_overflow
func.func @arith.floordivsi_by_one_overflow() -> i64 {
  %neg_one = arith.constant -1 : i64
  %min_int = arith.constant -9223372036854775808 : i64
  // CHECK: arith.floordivsi
  %poision = arith.floordivsi %min_int, %neg_one : i64
  return %poision : i64
}

// -----

// CHECK-LABEL: func @arith.ceildivsi_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @arith.ceildivsi_by_one(%arg0: i32) -> (i32) {
  %c1 = arith.constant 1 : i32
  %res = arith.ceildivsi %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @tensor_arith.ceildivsi_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @tensor_arith.ceildivsi_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = arith.constant dense<1> : tensor<4x5xi32>
  %res = arith.ceildivsi %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// -----

// CHECK-LABEL: func @arith.ceildivui_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @arith.ceildivui_by_one(%arg0: i32) -> (i32) {
  %c1 = arith.constant 1 : i32
  %res = arith.ceildivui %arg0, %c1 : i32
  // CHECK: return %[[ARG]]
  return %res : i32
}

// CHECK-LABEL: func @tensor_arith.ceildivui_by_one
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
func.func @tensor_arith.ceildivui_by_one(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c1 = arith.constant dense<1> : tensor<4x5xi32>
  %res = arith.ceildivui %arg0, %c1 : tensor<4x5xi32>
  // CHECK: return %[[ARG]]
  return %res : tensor<4x5xi32>
}

// -----

// CHECK-LABEL: func @memref_cast_folding_subview
func.func @memref_cast_folding_subview(%arg0: memref<4x5xf32>, %i: index) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>) {
  %0 = memref.cast %arg0 : memref<4x5xf32> to memref<?x?xf32>
  // CHECK-NEXT: memref.subview %{{.*}}: memref<4x5xf32>
  %1 = memref.subview %0[%i, %i][%i, %i][%i, %i]: memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %1: memref<?x?xf32, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: func @memref_cast_folding_subview_static(
func.func @memref_cast_folding_subview_static(%V: memref<16x16xf32>, %a: index, %b: index)
  -> memref<3x4xf32, strided<[?, 1], offset: ?>>
{
  %0 = memref.cast %V : memref<16x16xf32> to memref<?x?xf32>
  %1 = memref.subview %0[0, 0][3, 4][1, 1] : memref<?x?xf32> to memref<3x4xf32, strided<[?, 1], offset: ?>>

  // CHECK:  memref.subview{{.*}}: memref<16x16xf32> to memref<3x4xf32, strided<[16, 1]>>
  return %1: memref<3x4xf32, strided<[?, 1], offset: ?>>
}

// -----

// CHECK-LABEL: func @slice
// CHECK-SAME: %[[ARG0:[0-9a-z]*]]: index, %[[ARG1:[0-9a-z]*]]: index
func.func @slice(%t: tensor<8x16x4xf32>, %arg0 : index, %arg1 : index)
  -> tensor<?x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index

  // CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [7, 11, 2] [1, 1, 1] :
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<7x11x2xf32>
  // tensor.cast gets folded away in consumer.
  //  CHECK-NOT: tensor.cast
  %1 = tensor.extract_slice %t[%c0, %c0, %c0] [%c7, %c11, %c2] [%c1, %c1, %c1]
    : tensor<8x16x4xf32> to tensor<?x?x?xf32>

  // Test: slice with one dynamic operand can also be folded.
  // CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [2, %[[ARG0]], 2] [1, 1, 1] :
  // CHECK-SAME: tensor<7x11x2xf32> to tensor<2x?x2xf32>
  // CHECK: tensor.cast %{{.*}} : tensor<2x?x2xf32> to tensor<?x?x?xf32>
  %2 = tensor.extract_slice %1[%c0, %c0, %c0] [%c2, %arg0, %c2] [%c1, %c1, %c1]
    : tensor<?x?x?xf32> to tensor<?x?x?xf32>

  return %2 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @fold_trunci
// CHECK-SAME:    (%[[ARG0:[0-9a-z]*]]: i1)
func.func @fold_trunci(%arg0: i1) -> i1 attributes {} {
  // CHECK-NEXT: return %[[ARG0]] : i1
  %0 = arith.extui %arg0 : i1 to i8
  %1 = arith.trunci %0 : i8 to i1
  return %1 : i1
}

// -----

// CHECK-LABEL: func @fold_trunci_vector
// CHECK-SAME:    (%[[ARG0:[0-9a-z]*]]: vector<4xi1>)
func.func @fold_trunci_vector(%arg0: vector<4xi1>) -> vector<4xi1> attributes {} {
  // CHECK-NEXT: return %[[ARG0]] : vector<4xi1>
  %0 = arith.extui %arg0 : vector<4xi1> to vector<4xi8>
  %1 = arith.trunci %0 : vector<4xi8> to vector<4xi1>
  return %1 : vector<4xi1>
}

// -----

// CHECK-LABEL: func @fold_trunci
// CHECK-SAME:    (%[[ARG0:[0-9a-z]*]]: i1)
func.func @fold_trunci(%arg0: i1) -> i2 attributes {} {
  // CHECK-NEXT: %[[RES:[0-9a-z]*]] = arith.extui %[[ARG0]] : i1 to i2
  // CHECK-NEXT: return %[[RES]] : i2
  %0 = arith.extui %arg0 : i1 to i8
  %1 = arith.trunci %0 : i8 to i2
  return %1 : i2
}

// -----

// CHECK-LABEL: func @fold_trunci_vector
// CHECK-SAME:    (%[[ARG0:[0-9a-z]*]]: vector<4xi1>)
func.func @fold_trunci_vector(%arg0: vector<4xi1>) -> vector<4xi2> attributes {} {
  // CHECK-NEXT: %[[RES:[0-9a-z]*]] = arith.extui %[[ARG0]] : vector<4xi1> to vector<4xi2>
  // CHECK-NEXT: return %[[RES]] : vector<4xi2>
  %0 = arith.extui %arg0 : vector<4xi1> to vector<4xi8>
  %1 = arith.trunci %0 : vector<4xi8> to vector<4xi2>
  return %1 : vector<4xi2>
}

// -----

// CHECK-LABEL: func @fold_trunci_sexti
// CHECK-SAME:    (%[[ARG0:[0-9a-z]*]]: i1)
func.func @fold_trunci_sexti(%arg0: i1) -> i1 attributes {} {
  // CHECK-NEXT: return %[[ARG0]] : i1
  %0 = arith.extsi %arg0 : i1 to i8
  %1 = arith.trunci %0 : i8 to i1
  return %1 : i1
}

// CHECK-LABEL: func @simple_clone_elimination
func.func @simple_clone_elimination() -> memref<5xf32> {
  %ret = memref.alloc() : memref<5xf32>
  %temp = bufferization.clone %ret : memref<5xf32> to memref<5xf32>
  memref.dealloc %temp : memref<5xf32>
  return %ret : memref<5xf32>
}
// CHECK-NEXT: %[[ret:.*]] = memref.alloc()
// CHECK-NOT: %{{.*}} = bufferization.clone
// CHECK-NOT: memref.dealloc %{{.*}}
// CHECK: return %[[ret]]

// -----

// CHECK-LABEL: func @clone_loop_alloc
func.func @clone_loop_alloc(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2xf32>, %arg4: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  memref.dealloc %0 : memref<2xf32>
  %1 = bufferization.clone %arg3 : memref<2xf32> to memref<2xf32>
  %2 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %1) -> (memref<2xf32>) {
    %3 = arith.cmpi eq, %arg5, %arg1 : index
    memref.dealloc %arg6 : memref<2xf32>
    %4 = memref.alloc() : memref<2xf32>
    %5 = bufferization.clone %4 : memref<2xf32> to memref<2xf32>
    memref.dealloc %4 : memref<2xf32>
    %6 = bufferization.clone %5 : memref<2xf32> to memref<2xf32>
    memref.dealloc %5 : memref<2xf32>
    scf.yield %6 : memref<2xf32>
  }
  memref.copy %2, %arg4 : memref<2xf32> to memref<2xf32>
  memref.dealloc %2 : memref<2xf32>
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = bufferization.clone
// CHECK-NEXT: %[[ALLOC1:.*]] = scf.for
// CHECK-NEXT: memref.dealloc
// CHECK-NEXT: %[[ALLOC2:.*]] = memref.alloc
// CHECK-NEXT: scf.yield %[[ALLOC2]]
// CHECK: memref.copy %[[ALLOC1]]
// CHECK-NEXT: memref.dealloc %[[ALLOC1]]

// -----

// CHECK-LABEL: func @clone_nested_region
func.func @clone_nested_region(%arg0: index, %arg1: index, %arg2: index) -> memref<?x?xf32> {
  %cmp = arith.cmpi eq, %arg0, %arg1 : index
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  %1 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    %3 = scf.if %cmp -> (memref<?x?xf32>) {
      %9 = bufferization.clone %1 : memref<?x?xf32> to memref<?x?xf32>
      scf.yield %9 : memref<?x?xf32>
    } else {
      %7 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
      %10 = bufferization.clone %7 : memref<?x?xf32> to memref<?x?xf32>
      memref.dealloc %7 : memref<?x?xf32>
      scf.yield %10 : memref<?x?xf32>
    }
    %6 = bufferization.clone %3 : memref<?x?xf32> to memref<?x?xf32>
    memref.dealloc %3 : memref<?x?xf32>
    scf.yield %6 : memref<?x?xf32>
  } else {
    %3 = memref.alloc(%arg1, %arg1) : memref<?x?xf32>
    %6 = bufferization.clone %3 : memref<?x?xf32> to memref<?x?xf32>
    memref.dealloc %3 : memref<?x?xf32>
    scf.yield %6 : memref<?x?xf32>
  }
  memref.dealloc %1 : memref<?x?xf32>
  return %2 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC1:.*]] = memref.alloc
// CHECK-NEXT: %[[ALLOC2:.*]] = scf.if
// CHECK-NEXT: %[[ALLOC3_1:.*]] = scf.if
// CHECK-NEXT: %[[ALLOC4_1:.*]] = bufferization.clone %[[ALLOC1]]
// CHECK-NEXT: scf.yield %[[ALLOC4_1]]
//      CHECK: %[[ALLOC4_2:.*]] = memref.alloc
// CHECK-NEXT: scf.yield %[[ALLOC4_2]]
//      CHECK: scf.yield %[[ALLOC3_1]]
//      CHECK: %[[ALLOC3_2:.*]] = memref.alloc
// CHECK-NEXT: scf.yield %[[ALLOC3_2]]
//      CHECK: memref.dealloc %[[ALLOC1]]
// CHECK-NEXT: return %[[ALLOC2]]

// -----

// CHECK-LABEL: func @test_materialize_failure
func.func @test_materialize_failure() -> i64 {
  %const = index.constant 1234
  // Cannot materialize this castu's output constant.
  // CHECK: index.castu
  %u = index.castu %const : index to i64
  return %u: i64
}
