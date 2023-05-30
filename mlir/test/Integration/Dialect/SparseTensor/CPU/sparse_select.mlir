// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>
#CSR = #sparse_tensor.encoding<{lvlTypes = ["dense", "compressed"]}>
#CSC = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>

//
// Traits for tensor operations.
//
#trait_vec_select = {
  indexing_maps = [
    affine_map<(i) -> (i)>, // A
    affine_map<(i) -> (i)>  // C (out)
  ],
  iterator_types = ["parallel"]
}

#trait_mat_select = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

module {
  func.func @vecSelect(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %cf1 = arith.constant 1.0 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d0): tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec_select
      ins(%arga: tensor<?xf64, #SparseVector>)
      outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64):
          %1 = sparse_tensor.select %a : f64 {
              ^bb0(%x: f64):
                %keep = arith.cmpf "oge", %x, %cf1 : f64
                sparse_tensor.yield %keep : i1
            }
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  func.func @matUpperTriangle(%arga: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #CSR>
    %xv = bufferization.alloc_tensor(%d0, %d1): tensor<?x?xf64, #CSR>
    %0 = linalg.generic #trait_mat_select
      ins(%arga: tensor<?x?xf64, #CSR>)
      outs(%xv: tensor<?x?xf64, #CSR>) {
        ^bb(%a: f64, %b: f64):
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %1 = sparse_tensor.select %a : f64 {
              ^bb0(%x: f64):
                %keep = arith.cmpi "ugt", %col, %row : index
                sparse_tensor.yield %keep : i1
            }
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #CSR>
    return %0 : tensor<?x?xf64, #CSR>
  }

  // Dumps a sparse vector of type f64.
  func.func @dump_vec(%arg0: tensor<?xf64, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<8xf64>
    vector.print %1 : vector<8xf64>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xf64, #SparseVector> to tensor<?xf64>
    %2 = vector.transfer_read %dv[%c0], %d0: tensor<?xf64>, vector<16xf64>
    vector.print %2 : vector<16xf64>
    return
  }

  // Dump a sparse matrix.
  func.func @dump_mat(%arg0: tensor<?x?xf64, #CSR>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #CSR> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<16xf64>
    vector.print %1 : vector<16xf64>
    %dm = sparse_tensor.convert %arg0 : tensor<?x?xf64, #CSR> to tensor<?x?xf64>
    %2 = vector.transfer_read %dm[%c0, %c0], %d0: tensor<?x?xf64>, vector<5x5xf64>
    vector.print %2 : vector<5x5xf64>
    return
  }

  // Driver method to call and verify vector kernels.
  func.func @entry() {
    %c0 = arith.constant 0 : index

    // Setup sparse matrices.
    %v1 = arith.constant sparse<
        [ [1], [3], [5], [7], [9] ],
        [ 1.0, 2.0, -4.0, 0.0, 5.0 ]
    > : tensor<10xf64>
    %m1 = arith.constant sparse<
        [ [0, 3], [1, 4], [2, 1], [2, 3], [3, 3], [3, 4], [4, 2] ],
        [ 1., 2., 3., 4., 5., 6., 7.]
    > : tensor<5x5xf64>
    %sv1 = sparse_tensor.convert %v1 : tensor<10xf64> to tensor<?xf64, #SparseVector>
    %sm1 = sparse_tensor.convert %m1 : tensor<5x5xf64> to tensor<?x?xf64, #CSR>

    // Call sparse matrix kernels.
    %1 = call @vecSelect(%sv1) : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %2 = call @matUpperTriangle(%sm1) : (tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR>

    //
    // Verify the results.
    //
    // CHECK:      ( 1, 2, -4, 0, 5, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 2, 0, -4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 0, 0, 0, 1, 0 ), ( 0, 0, 0, 0, 2 ), ( 0, 3, 0, 4, 0 ), ( 0, 0, 0, 5, 6 ), ( 0, 0, 7, 0, 0 ) )
    // CHECK-NEXT: ( 1, 2, 5, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 1, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 0, 0, 0, 1, 0 ), ( 0, 0, 0, 0, 2 ), ( 0, 0, 0, 4, 0 ), ( 0, 0, 0, 0, 6 ), ( 0, 0, 0, 0, 0 ) )
    //
    call @dump_vec(%sv1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_mat(%sm1) : (tensor<?x?xf64, #CSR>) -> ()
    call @dump_vec(%1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_mat(%2) : (tensor<?x?xf64, #CSR>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?x?xf64, #CSR>
    return
  }
}
