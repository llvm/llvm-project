// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#SparseMatrix = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>

module @func_sparse.2 {
  // Do elementwise x+1 when true, x-1 when false
  func.func public @condition(%cond: i1, %arg0: tensor<2x3x4xf64, #SparseMatrix>) -> tensor<2x3x4xf64, #SparseMatrix> {
    %1 = scf.if %cond -> (tensor<2x3x4xf64, #SparseMatrix>) {
      %cst_2 = arith.constant dense<1.000000e+00> : tensor<f64>
      %cst_3 = arith.constant dense<1.000000e+00> : tensor<2x3x4xf64>
      %2 = bufferization.alloc_tensor() : tensor<2x3x4xf64, #SparseMatrix>
      %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %cst_3 : tensor<2x3x4xf64, #SparseMatrix>, tensor<2x3x4xf64>)
        outs(%2 : tensor<2x3x4xf64, #SparseMatrix>) {
          ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
            %4 = arith.subf %arg1, %arg2 : f64
            linalg.yield %4 : f64
          } -> tensor<2x3x4xf64, #SparseMatrix>
        scf.yield %3 : tensor<2x3x4xf64, #SparseMatrix>
    } else {
      %cst_2 = arith.constant dense<1.000000e+00> : tensor<f64>
      %cst_3 = arith.constant dense<1.000000e+00> : tensor<2x3x4xf64>
      %2 = bufferization.alloc_tensor() : tensor<2x3x4xf64, #SparseMatrix>
      %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %cst_3 : tensor<2x3x4xf64, #SparseMatrix>, tensor<2x3x4xf64>)
        outs(%2 : tensor<2x3x4xf64, #SparseMatrix>) {
          ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
            %4 = arith.addf %arg1, %arg2 : f64
            linalg.yield %4 : f64
          } -> tensor<2x3x4xf64, #SparseMatrix>
        scf.yield %3 : tensor<2x3x4xf64, #SparseMatrix>
    }
    return %1 : tensor<2x3x4xf64, #SparseMatrix>
  }

  func.func @dump(%arg0: tensor<2x3x4xf64, #SparseMatrix>) {
    %d0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %dm = sparse_tensor.convert %arg0 : tensor<2x3x4xf64, #SparseMatrix> to tensor<2x3x4xf64>
    %0 = vector.transfer_read %dm[%c0, %c0, %c0], %d0: tensor<2x3x4xf64>, vector<2x3x4xf64>
    vector.print %0 : vector<2x3x4xf64>
    return
  }

  func.func public @entry() {
    %src = arith.constant dense<[
     [  [  1.0,  2.0,  3.0,  4.0 ],
        [  5.0,  6.0,  7.0,  8.0 ],
        [  9.0, 10.0, 11.0, 12.0 ] ],
     [  [ 13.0, 14.0, 15.0, 16.0 ],
        [ 17.0, 18.0, 19.0, 20.0 ],
        [ 21.0, 22.0, 23.0, 24.0 ] ]
    ]> : tensor<2x3x4xf64>

    %t = arith.constant 1 : i1
    %f = arith.constant 0 : i1

    %sm = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #SparseMatrix>

    %sm_t = call @condition(%t, %sm) : (i1, tensor<2x3x4xf64, #SparseMatrix>) -> tensor<2x3x4xf64, #SparseMatrix>
    %sm_f = call @condition(%f, %sm) : (i1, tensor<2x3x4xf64, #SparseMatrix>) -> tensor<2x3x4xf64, #SparseMatrix>

    // CHECK:      ( ( ( 0, 1, 2, 3 ), ( 4, 5, 6, 7 ), ( 8, 9, 10, 11 ) ), ( ( 12, 13, 14, 15 ), ( 16, 17, 18, 19 ), ( 20, 21, 22, 23 ) ) )
    // CHECK-NEXT: ( ( ( 2, 3, 4, 5 ), ( 6, 7, 8, 9 ), ( 10, 11, 12, 13 ) ), ( ( 14, 15, 16, 17 ), ( 18, 19, 20, 21 ), ( 22, 23, 24, 25 ) ) )
    call @dump(%sm_t) : (tensor<2x3x4xf64, #SparseMatrix>) -> ()
    call @dump(%sm_f) : (tensor<2x3x4xf64, #SparseMatrix>) -> ()

    bufferization.dealloc_tensor %sm : tensor<2x3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %sm_t : tensor<2x3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %sm_f : tensor<2x3x4xf64, #SparseMatrix>
    return
  }
}