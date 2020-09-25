// RUN: mlir-opt -normalize-memrefs %s | FileCheck %s

// For all these cases, we test if MemRefs Normalization works with the test
// operations.
// * test.op_norm: this operation has the MemRefsNormalizable attribute. The tests
//   that include this operation are constructed so that the normalization should
//   happen.
// * test_op_nonnorm: this operation does not have the MemRefsNormalization
//   attribute. The tests that include this operation are contructed so that the
//    normalization should not happen.

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 32, d3 floordiv 64, d2 mod 32, d3 mod 64)>

// Test with op_norm and maps in arguments and in the operations in the function.

// CHECK-LABEL: test_norm
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: memref<1x16x1x1x32x64xf32>)
func @test_norm(%arg0 : memref<1x16x14x14xf32, #map0>) -> () {
    %0 = alloc() : memref<1x16x14x14xf32, #map0>
    "test.op_norm"(%arg0, %0) : (memref<1x16x14x14xf32, #map0>, memref<1x16x14x14xf32, #map0>) -> ()
    dealloc %0 :  memref<1x16x14x14xf32, #map0>

    // CHECK: %[[v0:[a-z0-9]*]] = alloc() : memref<1x16x1x1x32x64xf32>
    // CHECK: "test.op_norm"(%[[ARG0]], %[[v0]]) : (memref<1x16x1x1x32x64xf32>, memref<1x16x1x1x32x64xf32>) -> ()
    // CHECK: dealloc %[[v0]] : memref<1x16x1x1x32x64xf32>
    return
}

// Same test with op_nonnorm, with maps in the argmentets and the operations in the function.

// CHECK-LABEL: test_nonnorm
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: memref<1x16x14x14xf32, #map0>)
func @test_nonnorm(%arg0 : memref<1x16x14x14xf32, #map0>) -> () {
    %0 = alloc() : memref<1x16x14x14xf32, #map0>
    "test.op_nonnorm"(%arg0, %0) : (memref<1x16x14x14xf32, #map0>, memref<1x16x14x14xf32, #map0>) -> ()
    dealloc %0 :  memref<1x16x14x14xf32, #map0>

    // CHECK: %[[v0:[a-z0-9]*]] = alloc() : memref<1x16x14x14xf32, #map0>
    // CHECK: "test.op_nonnorm"(%[[ARG0]], %[[v0]]) : (memref<1x16x14x14xf32, #map0>, memref<1x16x14x14xf32, #map0>) -> ()
    // CHECK: dealloc %[[v0]] : memref<1x16x14x14xf32, #map0>
    return
}

// Test with op_norm, with maps in the operations in the function.

// CHECK-LABEL: test_norm_mix
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: memref<1x16x1x1x32x64xf32>
func @test_norm_mix(%arg0 : memref<1x16x1x1x32x64xf32>) -> () {
    %0 = alloc() : memref<1x16x14x14xf32, #map0>
    "test.op_norm"(%arg0, %0) : (memref<1x16x1x1x32x64xf32>, memref<1x16x14x14xf32, #map0>) -> ()
    dealloc %0 :  memref<1x16x14x14xf32, #map0>

    // CHECK: %[[v0:[a-z0-9]*]] = alloc() : memref<1x16x1x1x32x64xf32>
    // CHECK: "test.op_norm"(%[[ARG0]], %[[v0]]) : (memref<1x16x1x1x32x64xf32>, memref<1x16x1x1x32x64xf32>) -> ()
    // CHECK: dealloc %[[v0]] : memref<1x16x1x1x32x64xf32>
    return
}

// Test with maps in load and store ops.

#map_tile = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 32, d3 floordiv 32, d2 mod 32, d3 mod 32)>

// CHECK-LABEL: test_load_store
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: memref<1x16x14x14xf32>
func @test_load_store(%arg0 : memref<1x16x14x14xf32>) -> () {
    %0 = alloc() : memref<1x16x14x14xf32, #map_tile>
    // CHECK: %[[v0:[a-z0-9]*]] = alloc() : memref<1x16x1x1x32x32xf32>
    %1 = alloc() : memref<1x16x14x14xf32>
    // CHECK: %[[v1:[a-z0-9]*]] = alloc() : memref<1x16x14x14xf32>
    "test.op_norm"(%0, %1) : (memref<1x16x14x14xf32, #map_tile>, memref<1x16x14x14xf32>) -> ()
    // CHECK: "test.op_norm"(%[[v0]], %[[v1]]) : (memref<1x16x1x1x32x32xf32>, memref<1x16x14x14xf32>) -> ()
    %cst = constant 3.0 : f32
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 16 {
        affine.for %k = 0 to 14 {
          affine.for %l = 0 to 14 {
            %2 = load %1[%i, %j, %k, %l] : memref<1x16x14x14xf32>
            // CHECK: memref<1x16x14x14xf32>
            %3 = addf %2, %cst : f32
            store %3, %arg0[%i, %j, %k, %l] : memref<1x16x14x14xf32>
            // CHECK: memref<1x16x14x14xf32>
          }
        }
      }
    }
    dealloc %0 :  memref<1x16x14x14xf32, #map_tile>
    // CHECK: dealloc %[[v0]] : memref<1x16x1x1x32x32xf32>
    dealloc %1 :  memref<1x16x14x14xf32>
    // CHECK: dealloc %[[v1]] : memref<1x16x14x14xf32>
    return
}

// Test with an arbitrary op that references the function symbol.

"test.op_funcref"() {func = @test_norm_mix} : () -> ()
