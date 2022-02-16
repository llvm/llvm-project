// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s -sparse-compiler="vectorization-strategy=2 vl=8" | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SM = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

#trait_sampled_dense_dense = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j)>,  // S
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += S(i,j) SUM_k A(i,k) B(k,j)"
}

#trait_matmul = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d1, d0)>,
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>
  ],
  iterator_types = ["reduction", "parallel", "parallel"]
}

#trait_scale = {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel"]
}

//
// Integration test for sampled dense dense matmul fusion.
//
module {
  //
  // A kernel that computes a direct sampled matrix matrix multiplication.
  //
  func @sampled_dd(%args: tensor<8x8xf64, #SM>,
                   %arga: tensor<8x8xf64>,
                   %argb: tensor<8x8xf64>) -> tensor<8x8xf64> {
    %1 = arith.constant dense<0.0> : tensor<8x8xf64>
    %2 = linalg.generic #trait_sampled_dense_dense
      ins(%args, %arga, %argb: tensor<8x8xf64, #SM>,
                               tensor<8x8xf64>, tensor<8x8xf64>)
      outs(%1: tensor<8x8xf64>) {
        ^bb(%s: f64, %a: f64, %b: f64, %x: f64):
          %p = arith.mulf %a, %b : f64
          %q = arith.mulf %s, %p : f64
          %r = arith.addf %x, %q : f64
          linalg.yield %r : f64
    } -> tensor<8x8xf64>
    return %2 : tensor<8x8xf64>
  }

  //
  // A kernel that computes an unfused sampled matrix matrix multiplication.
  //
  func @sampled_dd_unfused(%args: tensor<8x8xf64, #SM>,
                           %arga: tensor<8x8xf64>,
                           %argb: tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8x8xf64>) {
    %1 = arith.constant dense<0.0> : tensor<8x8xf64>
    %2 = linalg.generic #trait_matmul
      ins(%arga, %argb : tensor<8x8xf64>, tensor<8x8xf64>)
      outs(%1 : tensor<8x8xf64>) {
        ^bb0(%a: f64, %b: f64, %x: f64):
          %p = arith.mulf %a, %b : f64
          %q = arith.addf %x, %p : f64
          linalg.yield %q : f64
    } -> tensor<8x8xf64>

    %3 = arith.constant dense<0.0> : tensor<8x8xf64>
    %4 = linalg.generic #trait_scale
      ins(%2, %args : tensor<8x8xf64>, tensor<8x8xf64, #SM>)
      outs(%3 : tensor<8x8xf64>) {
        ^bb0(%t: f64, %s: f64, %x: f64):
          %r = arith.mulf %t, %s : f64
          linalg.yield %r : f64
    } -> tensor<8x8xf64>

    return %4, %2 : tensor<8x8xf64>, tensor<8x8xf64>
  }

  //
  // Main driver.
  //
  func @entry() {
    %d0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index

    %t = arith.constant sparse<[[0, 0], [7,7]], [1.0, 2.0]>
       : tensor<8x8xf64>
    %s = sparse_tensor.convert %t
       : tensor<8x8xf64> to tensor<8x8xf64, #SM>

    %a = arith.constant dense<3.0> : tensor<8x8xf64>
    %b = arith.constant dense<4.0> : tensor<8x8xf64>

    // Call the kernels.
    %0 = call @sampled_dd(%s, %a, %b)
      : (tensor<8x8xf64, #SM>,
         tensor<8x8xf64>, tensor<8x8xf64>) -> tensor<8x8xf64>
    %1, %2 = call @sampled_dd_unfused(%s, %a, %b)
      : (tensor<8x8xf64, #SM>,
         tensor<8x8xf64>, tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8x8xf64>)

    // Verify the outputs.
    //
    // CHECK:    ( ( 96, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 192 ) )
    //
    // CHECK:    ( ( 96, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 192 ) )
    //
    %m0 = bufferization.to_memref %0 : memref<8x8xf64>
    %m1 = bufferization.to_memref %1 : memref<8x8xf64>
    %m2 = bufferization.to_memref %2 : memref<8x8xf64>
    %v0 = vector.transfer_read %m0[%c0, %c0], %d0
        : memref<8x8xf64>, vector<8x8xf64>
    %v1 = vector.transfer_read %m1[%c0, %c0], %d0
        : memref<8x8xf64>, vector<8x8xf64>
    vector.print %v0 : vector<8x8xf64>
    vector.print %v1 : vector<8x8xf64>

    // Release the resources.
    sparse_tensor.release %s : tensor<8x8xf64, #SM>
    memref.dealloc %m0 : memref<8x8xf64>
    memref.dealloc %m1 : memref<8x8xf64>
    memref.dealloc %m2 : memref<8x8xf64>

    return
  }
}
