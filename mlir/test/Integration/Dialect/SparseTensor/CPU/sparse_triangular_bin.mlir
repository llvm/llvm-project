// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseMatrix = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

#trait_op = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (i,j)>, // B
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel","parallel"],
  doc = "X(i,j) = A(i,j) OP B(i,j)"
}

module {
  // Performs triangular add/sub operation (using semi-ring binary op).
  func.func @triangular(%A: tensor<4x4xf64, #SparseMatrix>,
                        %B: tensor<4x4xf64, #SparseMatrix>) -> tensor<4x4xf64, #SparseMatrix> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #SparseMatrix>
    %0 = linalg.generic #trait_op
      ins(%A, %B: tensor<4x4xf64, #SparseMatrix>,
                  tensor<4x4xf64, #SparseMatrix>)
      outs(%C: tensor<4x4xf64, #SparseMatrix>) {
        ^bb0(%a: f64, %b: f64, %c: f64) :
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %result = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpi "uge", %col, %row : index
                %upperTriangleResult = arith.addf %x, %y : f64
                %lowerTriangleResult = arith.subf %x, %y : f64
                %ret = arith.select %cmp, %upperTriangleResult, %lowerTriangleResult : f64
                sparse_tensor.yield %ret : f64
            }
            left=identity
            right={
              ^bb0(%y: f64):
                %cmp = arith.cmpi "uge", %col, %row : index
                %lowerTriangleResult = arith.negf %y : f64
                %ret = arith.select %cmp, %y, %lowerTriangleResult : f64
                sparse_tensor.yield %ret : f64
            }
          linalg.yield %result : f64
      } -> tensor<4x4xf64, #SparseMatrix>
    return %0 : tensor<4x4xf64, #SparseMatrix>
  }

  // Driver method to call and verify triangular kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %du = arith.constant -1.0 : f64

    %am = arith.constant dense<
      [ [ 1.0, 0.0, 3.0, 0.0],
        [ 0.0, 2.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 4.0],
        [ 3.0, 4.0, 0.0, 0.0] ]> : tensor<4x4xf64>
    %bm = arith.constant dense<
      [ [ 1.0, 0.0, 1.0, 1.0],
        [ 0.0, 0.5, 0.0, 0.0],
        [ 1.0, 5.0, 2.0, 0.0],
        [ 2.0, 0.0, 0.0, 0.0] ]> : tensor<4x4xf64>

    %a = sparse_tensor.convert %am : tensor<4x4xf64> to tensor<4x4xf64, #SparseMatrix>
    %b = sparse_tensor.convert %bm : tensor<4x4xf64> to tensor<4x4xf64, #SparseMatrix>
    %0 = call @triangular(%a, %b) : (tensor<4x4xf64, #SparseMatrix>,
                                     tensor<4x4xf64, #SparseMatrix>) -> tensor<4x4xf64, #SparseMatrix>

    //
    // Verify the results.
    //
    // CHECK:    ( ( 2, 0, 4, 1 ), ( 0, 2.5, 0, 0 ), ( -1, -5, 2, 4 ), ( 1, 4, 0, 0 ) )
    // CHECK-NEXT: ( 2, 4, 1, 2.5, -1, -5, 2, 4, 1, 4, -1, -1, -1, -1, -1, -1 )
    //
    %c = sparse_tensor.convert %0 : tensor<4x4xf64, #SparseMatrix> to tensor<4x4xf64>
    %m = bufferization.to_memref %c : memref<4x4xf64>
    %v = vector.transfer_read %m[%c0, %c0], %du: memref<4x4xf64>, vector<4x4xf64>
    vector.print %v : vector<4x4xf64>
    %1 = sparse_tensor.values %0 : tensor<4x4xf64, #SparseMatrix> to memref<?xf64>
    %2 = vector.transfer_read %1[%c0], %du: memref<?xf64>, vector<16xf64>
    vector.print %2 : vector<16xf64>

    // Release the resources.
    memref.dealloc %m : memref<4x4xf64>
    sparse_tensor.release %a : tensor<4x4xf64, #SparseMatrix>
    sparse_tensor.release %b : tensor<4x4xf64, #SparseMatrix>
    sparse_tensor.release %0 : tensor<4x4xf64, #SparseMatrix>
    return
  }
}
