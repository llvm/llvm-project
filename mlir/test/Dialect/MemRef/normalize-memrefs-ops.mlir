// RUN: mlir-opt -normalize-memrefs %s | FileCheck %s

// For all these cases, we test if MemRefs Normalization works with the test
// operations.
// * test.op_norm: this operation has the MemRefsNormalizable attribute. The tests
//   that include this operation are constructed so that the normalization should
//   happen.
// * test_op_nonnorm: this operation does not have the MemRefsNormalization
//   attribute. The tests that include this operation are constructed so that the
//    normalization should not happen.

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 32, d3 floordiv 64, d2 mod 32, d3 mod 64)>

// Test with op_norm and maps in arguments and in the operations in the function.

// CHECK-LABEL: test_norm
// CHECK-SAME: (%[[ARG0:.*]]: memref<1x16x1x1x32x64xf32>)
func.func @test_norm(%arg0 : memref<1x16x14x14xf32, #map0>) -> () {
    %0 = memref.alloc() : memref<1x16x14x14xf32, #map0>
    "test.op_norm"(%arg0, %0) : (memref<1x16x14x14xf32, #map0>, memref<1x16x14x14xf32, #map0>) -> ()
    memref.dealloc %0 :  memref<1x16x14x14xf32, #map0>

    // CHECK: %[[v0:.*]] = memref.alloc() : memref<1x16x1x1x32x64xf32>
    // CHECK: "test.op_norm"(%[[ARG0]], %[[v0]]) : (memref<1x16x1x1x32x64xf32>, memref<1x16x1x1x32x64xf32>) -> ()
    // CHECK: memref.dealloc %[[v0]] : memref<1x16x1x1x32x64xf32>
    return
}

// Same test with op_nonnorm, with maps in the arguments and the operations in the function.

// CHECK-LABEL: test_nonnorm
// CHECK-SAME: (%[[ARG0:.*]]: memref<1x16x14x14xf32, #[[MAP:.*]]>)
func.func @test_nonnorm(%arg0 : memref<1x16x14x14xf32, #map0>) -> () {
    %0 = memref.alloc() : memref<1x16x14x14xf32, #map0>
    "test.op_nonnorm"(%arg0, %0) : (memref<1x16x14x14xf32, #map0>, memref<1x16x14x14xf32, #map0>) -> ()
    memref.dealloc %0 :  memref<1x16x14x14xf32, #map0>

    // CHECK: %[[v0:.*]] = memref.alloc() : memref<1x16x14x14xf32, #[[MAP]]>
    // CHECK: "test.op_nonnorm"(%[[ARG0]], %[[v0]]) : (memref<1x16x14x14xf32, #[[MAP]]>, memref<1x16x14x14xf32, #[[MAP]]>) -> ()
    // CHECK: memref.dealloc %[[v0]] : memref<1x16x14x14xf32, #[[MAP]]>
    return
}

// Test with op_nonnorm whose memref map layouts are identity. This op_nonnorm
// does not block the normalization of other operations.

// CHECK-LABEL: test_nonnorm_identity_layout
// CHECK-SAME: (%[[ARG0:.*]]: memref<1x16x1x1x32x64xf32>)
func.func @test_nonnorm_identity_layout(%arg0 : memref<1x16x14x14xf32, #map0>) -> () {
    %0 = memref.alloc() : memref<1x16x14x14xf32>
    "test.op_nonnorm"(%0, %0) : (memref<1x16x14x14xf32>, memref<1x16x14x14xf32>) -> ()
    "test.op_norm"(%arg0, %0) : (memref<1x16x14x14xf32, #map0>, memref<1x16x14x14xf32>) -> ()
    memref.dealloc %0 :  memref<1x16x14x14xf32>

    // CHECK: %[[v0:.*]] = memref.alloc() : memref<1x16x14x14xf32>
    // CHECK: "test.op_nonnorm"(%[[v0]], %[[v0]]) : (memref<1x16x14x14xf32>, memref<1x16x14x14xf32>) -> ()
    // CHECK: "test.op_norm"(%[[ARG0]], %[[v0]]) : (memref<1x16x1x1x32x64xf32>, memref<1x16x14x14xf32>) -> ()
    // CHECK: memref.dealloc %[[v0]] : memref<1x16x14x14xf32>
    return
}

// Test with op_norm, with maps in the operations in the function.

// CHECK-LABEL: test_norm_mix
// CHECK-SAME: (%[[ARG0:.*]]: memref<1x16x1x1x32x64xf32>
func.func @test_norm_mix(%arg0 : memref<1x16x1x1x32x64xf32>) -> () {
    %0 = memref.alloc() : memref<1x16x14x14xf32, #map0>
    "test.op_norm"(%arg0, %0) : (memref<1x16x1x1x32x64xf32>, memref<1x16x14x14xf32, #map0>) -> ()
    memref.dealloc %0 :  memref<1x16x14x14xf32, #map0>

    // CHECK: %[[v0:.*]] = memref.alloc() : memref<1x16x1x1x32x64xf32>
    // CHECK: "test.op_norm"(%[[ARG0]], %[[v0]]) : (memref<1x16x1x1x32x64xf32>, memref<1x16x1x1x32x64xf32>) -> ()
    // CHECK: memref.dealloc %[[v0]] : memref<1x16x1x1x32x64xf32>
    return
}

// Test with maps in load and store ops.

#map_tile = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 32, d3 floordiv 32, d2 mod 32, d3 mod 32)>

// CHECK-LABEL: test_load_store
// CHECK-SAME: (%[[ARG0:.*]]: memref<1x16x14x14xf32>
func.func @test_load_store(%arg0 : memref<1x16x14x14xf32>) -> () {
    %0 = memref.alloc() : memref<1x16x14x14xf32, #map_tile>
    // CHECK: %[[v0:.*]] = memref.alloc() : memref<1x16x1x1x32x32xf32>
    %1 = memref.alloc() : memref<1x16x14x14xf32>
    // CHECK: %[[v1:.*]] = memref.alloc() : memref<1x16x14x14xf32>
    "test.op_norm"(%0, %1) : (memref<1x16x14x14xf32, #map_tile>, memref<1x16x14x14xf32>) -> ()
    // CHECK: "test.op_norm"(%[[v0]], %[[v1]]) : (memref<1x16x1x1x32x32xf32>, memref<1x16x14x14xf32>) -> ()
    %cst = arith.constant 3.0 : f32
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 16 {
        affine.for %k = 0 to 14 {
          affine.for %l = 0 to 14 {
            %2 = memref.load %1[%i, %j, %k, %l] : memref<1x16x14x14xf32>
            // CHECK: memref<1x16x14x14xf32>
            %3 = arith.addf %2, %cst : f32
            memref.store %3, %arg0[%i, %j, %k, %l] : memref<1x16x14x14xf32>
            // CHECK: memref<1x16x14x14xf32>
          }
        }
      }
    }
    memref.dealloc %0 :  memref<1x16x14x14xf32, #map_tile>
    // CHECK: memref.dealloc %[[v0]] : memref<1x16x1x1x32x32xf32>
    memref.dealloc %1 :  memref<1x16x14x14xf32>
    // CHECK: memref.dealloc %[[v1]] : memref<1x16x14x14xf32>
    return
}

// Test with op_norm_ret, with maps in the results of normalizable operation.

// CHECK-LABEL: test_norm_ret
// CHECK-SAME: (%[[ARG0:.*]]: memref<1x16x1x1x32x32xf32>) -> (memref<1x16x1x1x32x32xf32>, memref<1x16x14x14xf32>) {
func.func @test_norm_ret(%arg0: memref<1x16x14x14xf32, #map_tile>) -> (memref<1x16x14x14xf32, #map_tile>, memref<1x16x14x14xf32>) {
    %0 = memref.alloc() : memref<1x16x14x14xf32, #map_tile>
    // CHECK-NEXT: %[[v0:.*]] = memref.alloc() : memref<1x16x1x1x32x32xf32>
    %1, %2 = "test.op_norm_ret"(%arg0) : (memref<1x16x14x14xf32, #map_tile>) -> (memref<1x16x14x14xf32, #map_tile>, memref<1x16x14x14xf32>)
    // CHECK-NEXT: %[[v1:.*]], %[[v2:.*]] = "test.op_norm_ret"
    // CHECK-SAME: (memref<1x16x1x1x32x32xf32>) -> (memref<1x16x1x1x32x32xf32>, memref<1x16x14x14xf32>)
    "test.op_norm"(%1, %0) : (memref<1x16x14x14xf32, #map_tile>, memref<1x16x14x14xf32, #map_tile>) -> ()
    // CHECK-NEXT: "test.op_norm"
    // CHECK-SAME: : (memref<1x16x1x1x32x32xf32>, memref<1x16x1x1x32x32xf32>) -> ()
    memref.dealloc %0 : memref<1x16x14x14xf32, #map_tile>
    // CHECK-NEXT: memref.dealloc %[[v0]] : memref<1x16x1x1x32x32xf32>
    return %1, %2 : memref<1x16x14x14xf32, #map_tile>, memref<1x16x14x14xf32>
    // CHECK-NEXT: return %[[v1]], %[[v2]] : memref<1x16x1x1x32x32xf32>, memref<1x16x14x14xf32>
}

// Test with an arbitrary op that references the function symbol.

"test.op_funcref"() {func = @test_norm_mix} : () -> ()


// -----

#map_1d_tile = affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)>

// Test with memref.reinterpret_cast

// CHECK-LABEL: test_norm_reinterpret_cast
// CHECK-SAME: (%[[ARG0:.*]]: memref<1x32xf32>) -> memref<3x1x1xf32> {
func.func @test_norm_reinterpret_cast(%arg0 : memref<3xf32, #map_1d_tile>) -> (memref<3x1x1xf32>) {
    %0 = memref.alloc() : memref<3xf32>
    "test.op_norm"(%arg0, %0) : (memref<3xf32, #map_1d_tile>, memref<3xf32>) -> ()
    %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [3, 1, 1], strides: [1, 1, 1] : memref<3xf32> to memref<3x1x1xf32>
    // CHECK: %[[v0:.*]] = memref.alloc() : memref<3xf32>
    // CHECK: "test.op_norm"(%[[ARG0]], %[[v0]]) : (memref<1x32xf32>, memref<3xf32>) -> ()
    // CHECK: memref.reinterpret_cast %[[v0]] to offset: [0], sizes: [3, 1, 1], strides: [1, 1, 1] : memref<3xf32> to memref<3x1x1xf32>
    return %1 : memref<3x1x1xf32>
}


// -----

// Test normalization of memrefs for prefetch.affine

// CHECK-LABEL: func.func @prefetch_normalize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xf32>) {
func.func @prefetch_normalize(%arg0: memref<512xf32, affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)>>) -> () {
  // CHECK: affine.for [[I_0_:%.+]] = 0 to 8 {
  affine.for %arg3 = 0 to 8  {
    // CHECK: affine.prefetch [[PARAM_0_]]{{.}}[[I_0_]] floordiv 32, [[I_0_]] mod 32], read, locality<3>, data : memref<16x32xf32>
    affine.prefetch %arg0[%arg3], read, locality<3>, data : memref<512xf32, affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)>>
  }
  return
}

#map_strided = affine_map<(d0, d1) -> (d0 * 7 + d1)>

// CHECK-LABEL: test_reinterpret_cast
func.func @test_reinterpret_cast(%arg0: memref<5x7xf32>, %arg1: memref<5x7xf32>, %arg2: memref<5x7xf32>) {
  %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [5, 7], strides: [7, 1] : memref<5x7xf32> to memref<5x7xf32, #map_strided>
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [35], strides: [1] : memref<5x7xf32> to memref<35xf32>
  affine.for %arg5 = 0 to 5 {
    affine.for %arg6 = 0 to 7 {
      %1 = affine.load %0[%arg5, %arg6] : memref<5x7xf32, #map_strided>
      // CHECK: affine.load %reinterpret_cast[%{{.*}} * 7 + %{{.*}}] : memref<35xf32>
      %2 = affine.load %arg1[%arg5, %arg6] : memref<5x7xf32>
      %3 = arith.subf %1, %2 : f32
      affine.store %3, %arg2[%arg5, %arg6] : memref<5x7xf32>
    }
  }
  return
}

// CHECK-LABEL: reinterpret_cast_non_zero_offset
func.func @reinterpret_cast_non_zero_offset(%arg0: index, %arg1: memref<1x10x17xi32, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1x10x17xi32, strided<[?, ?, ?], offset: ?>>, %arg3: memref<1x10x17xi32, strided<[?, ?, ?], offset: ?>>) -> (memref<1x5xf32, strided<[17, 1], offset: 27>>, memref<1x5xf32, strided<[17, 1], offset: 27>>, memref<2x17xf32>, memref<1x10x17xi32>, memref<1x10x17xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x10x17xi32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x17xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x10x17xf32>
  cf.br ^bb3
^bb3:  // pred: ^bb1
  // CHECK: %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [32], strides: [1] : memref<2x17xf32> to memref<32xf32>
  // CHECK: return %[[REINTERPRET_CAST]], %[[REINTERPRET_CAST]], %{{.*}}, %{{.*}}, %{{.*}} : memref<32xf32>, memref<32xf32>, memref<2x17xf32>, memref<1x10x17xi32>, memref<1x10x17xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc_0 to offset: [27], sizes: [1, 5], strides: [17, 1] : memref<2x17xf32> to memref<1x5xf32, strided<[17, 1], offset: 27>>
  return %reinterpret_cast, %reinterpret_cast, %alloc_0, %alloc, %alloc_1 : memref<1x5xf32, strided<[17, 1], offset: 27>>, memref<1x5xf32, strided<[17, 1], offset: 27>>, memref<2x17xf32>, memref<1x10x17xi32>, memref<1x10x17xf32>
}
