// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext --dlopen=%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#MAT_C_C = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>
#MAT_D_C = #sparse_tensor.encoding<{lvlTypes = ["dense", "compressed"]}>
#MAT_C_D = #sparse_tensor.encoding<{lvlTypes = ["compressed", "dense"]}>
#MAT_D_D = #sparse_tensor.encoding<{
  lvlTypes = ["dense", "dense"],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>

#MAT_C_C_P = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>

#MAT_C_D_P = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "dense" ],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>

#MAT_D_C_P = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>

module {
  func.func private @printMemrefF64(%ptr : tensor<*xf64>)
  func.func private @printMemref1dF64(%ptr : memref<?xf64>) attributes { llvm.emit_c_interface }

  //
  // Tests with permutation.
  //

  // Concats all sparse matrices (with different encodings) to a sparse matrix.
  func.func @concat_sparse_sparse_perm(%arg0: tensor<2x4xf64, #MAT_C_C_P>, %arg1: tensor<3x4xf64, #MAT_C_D>, %arg2: tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64, #MAT_C_C_P> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #MAT_C_C_P>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C> to tensor<9x4xf64, #MAT_C_C_P>
    return %0 : tensor<9x4xf64, #MAT_C_C_P>
  }

  // Concats all sparse matrices (with different encodings) to a dense matrix.
  func.func @concat_sparse_dense_perm(%arg0: tensor<2x4xf64, #MAT_C_C_P>, %arg1: tensor<3x4xf64, #MAT_C_D_P>, %arg2: tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #MAT_C_C_P>, tensor<3x4xf64, #MAT_C_D_P>, tensor<4x4xf64, #MAT_D_C> to tensor<9x4xf64>
    return %0 : tensor<9x4xf64>
  }

  // Concats mix sparse and dense matrices to a sparse matrix.
  func.func @concat_mix_sparse_perm(%arg0: tensor<2x4xf64>, %arg1: tensor<3x4xf64, #MAT_C_D_P>, %arg2: tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64, #MAT_C_C> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64>, tensor<3x4xf64, #MAT_C_D_P>, tensor<4x4xf64, #MAT_D_C> to tensor<9x4xf64, #MAT_C_C>
    return %0 : tensor<9x4xf64, #MAT_C_C>
  }

  // Concats mix sparse and dense matrices to a dense matrix.
  func.func @concat_mix_dense_perm(%arg0: tensor<2x4xf64>, %arg1: tensor<3x4xf64, #MAT_C_D>, %arg2: tensor<4x4xf64, #MAT_D_C_P>) -> tensor<9x4xf64> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C_P> to tensor<9x4xf64>
    return %0 : tensor<9x4xf64>
  }

  func.func @dump_mat_9x4(%A: tensor<9x4xf64, #MAT_C_C>) {
    %c = sparse_tensor.convert %A : tensor<9x4xf64, #MAT_C_C> to tensor<9x4xf64>
    %cu = tensor.cast %c : tensor<9x4xf64> to tensor<*xf64>
    call @printMemrefF64(%cu) : (tensor<*xf64>) -> ()

    %n = sparse_tensor.number_of_entries %A : tensor<9x4xf64, #MAT_C_C>
    vector.print %n : index

    %1 = sparse_tensor.values %A : tensor<9x4xf64, #MAT_C_C> to memref<?xf64>
    call @printMemref1dF64(%1) : (memref<?xf64>) -> ()

    return
  }

  func.func @dump_mat_perm_9x4(%A: tensor<9x4xf64, #MAT_C_C_P>) {
    %c = sparse_tensor.convert %A : tensor<9x4xf64, #MAT_C_C_P> to tensor<9x4xf64>
    %cu = tensor.cast %c : tensor<9x4xf64> to tensor<*xf64>
    call @printMemrefF64(%cu) : (tensor<*xf64>) -> ()

    %n = sparse_tensor.number_of_entries %A : tensor<9x4xf64, #MAT_C_C_P>
    vector.print %n : index

    %1 = sparse_tensor.values %A : tensor<9x4xf64, #MAT_C_C_P> to memref<?xf64>
    call @printMemref1dF64(%1) : (memref<?xf64>) -> ()

    return
  }

  func.func @dump_mat_dense_9x4(%A: tensor<9x4xf64>) {
    %u = tensor.cast %A : tensor<9x4xf64> to tensor<*xf64>
    call @printMemrefF64(%u) : (tensor<*xf64>) -> ()

    return
  }

  func.func @dump_mat_annotated_dense_9x4(%A: tensor<9x4xf64, #MAT_D_D>) {
    %n = sparse_tensor.number_of_entries %A : tensor<9x4xf64, #MAT_D_D>
    vector.print %n : index

    %1 = sparse_tensor.values %A : tensor<9x4xf64, #MAT_D_D> to memref<?xf64>
    call @printMemref1dF64(%1) : (memref<?xf64>) -> ()

    return
  }

  // Driver method to call and verify kernels.
  func.func @entry() {
    %m42 = arith.constant dense<
      [ [ 1.0, 0.0 ],
        [ 3.1, 0.0 ],
        [ 0.0, 2.0 ],
        [ 0.0, 0.0 ] ]> : tensor<4x2xf64>
    %m43 = arith.constant dense<
      [ [ 1.0, 0.0, 1.0 ],
        [ 1.0, 0.0, 0.5 ],
        [ 0.0, 0.0, 1.0 ],
        [ 5.0, 2.0, 0.0 ] ]> : tensor<4x3xf64>
    %m24 = arith.constant dense<
      [ [ 1.0, 0.0, 3.0, 0.0],
        [ 0.0, 2.0, 0.0, 0.0] ]> : tensor<2x4xf64>
    %m34 = arith.constant dense<
      [ [ 1.0, 0.0, 1.0, 1.0],
        [ 0.0, 0.5, 0.0, 0.0],
        [ 1.0, 5.0, 2.0, 0.0] ]> : tensor<3x4xf64>
    %m44 = arith.constant dense<
      [ [ 0.0, 0.0, 1.5, 1.0],
        [ 0.0, 3.5, 0.0, 0.0],
        [ 1.0, 5.0, 2.0, 0.0],
        [ 1.0, 0.5, 0.0, 0.0] ]> : tensor<4x4xf64>

    %sm24cc = sparse_tensor.convert %m24 : tensor<2x4xf64> to tensor<2x4xf64, #MAT_C_C>
    %sm34cd = sparse_tensor.convert %m34 : tensor<3x4xf64> to tensor<3x4xf64, #MAT_C_D>
    %sm44dc = sparse_tensor.convert %m44 : tensor<4x4xf64> to tensor<4x4xf64, #MAT_D_C>

    %sm24ccp = sparse_tensor.convert %m24 : tensor<2x4xf64> to tensor<2x4xf64, #MAT_C_C_P>
    %sm34cdp = sparse_tensor.convert %m34 : tensor<3x4xf64> to tensor<3x4xf64, #MAT_C_D_P>
    %sm44dcp = sparse_tensor.convert %m44 : tensor<4x4xf64> to tensor<4x4xf64, #MAT_D_C_P>

    // CHECK:      {{\[}}[1,   0,   3,   0],
    // CHECK-NEXT:  [0,   2,   0,   0],
    // CHECK-NEXT:  [1,   0,   1,   1],
    // CHECK-NEXT:  [0,   0.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [0,   0,   1.5,   1],
    // CHECK-NEXT:  [0,   3.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [1,   0.5,   0,   0]]
    // CHECK-NEXT: 18
    // CHECK:      [1,  1,  1,  1,  1,  2,  0.5,  5,  3.5,  5,  0.5,  3,  1,  2,  1.5,  2,  1,  1
    %4 = call @concat_sparse_sparse_perm(%sm24ccp, %sm34cd, %sm44dc)
               : (tensor<2x4xf64, #MAT_C_C_P>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64, #MAT_C_C_P>
    call @dump_mat_perm_9x4(%4) : (tensor<9x4xf64, #MAT_C_C_P>) -> ()

    // CHECK:      {{\[}}[1,   0,   3,   0],
    // CHECK-NEXT:  [0,   2,   0,   0],
    // CHECK-NEXT:  [1,   0,   1,   1],
    // CHECK-NEXT:  [0,   0.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [0,   0,   1.5,   1],
    // CHECK-NEXT:  [0,   3.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [1,   0.5,   0,   0]]
    %5 = call @concat_sparse_dense_perm(%sm24ccp, %sm34cdp, %sm44dc)
               : (tensor<2x4xf64, #MAT_C_C_P>, tensor<3x4xf64, #MAT_C_D_P>, tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64>
    call @dump_mat_dense_9x4(%5) : (tensor<9x4xf64>) -> ()

    // CHECK:      {{\[}}[1,   0,   3,   0],
    // CHECK-NEXT:  [0,   2,   0,   0],
    // CHECK-NEXT:  [1,   0,   1,   1],
    // CHECK-NEXT:  [0,   0.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [0,   0,   1.5,   1],
    // CHECK-NEXT:  [0,   3.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [1,   0.5,   0,   0]]
    // CHECK-NEXT: 18
    // CHECK:      [1,  3,  2,  1,  1,  1,  0.5,  1,  5,  2,  1.5,  1,  3.5,  1,  5,  2,  1,  0.5
    %6 = call @concat_mix_sparse_perm(%m24, %sm34cdp, %sm44dc)
               : (tensor<2x4xf64>, tensor<3x4xf64, #MAT_C_D_P>, tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64, #MAT_C_C>
    call @dump_mat_9x4(%6) : (tensor<9x4xf64, #MAT_C_C>) -> ()

    // CHECK:      {{\[}}[1,   0,   3,   0],
    // CHECK-NEXT:  [0,   2,   0,   0],
    // CHECK-NEXT:  [1,   0,   1,   1],
    // CHECK-NEXT:  [0,   0.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [0,   0,   1.5,   1],
    // CHECK-NEXT:  [0,   3.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [1,   0.5,   0,   0]]
    %7 = call @concat_mix_dense_perm(%m24, %sm34cd, %sm44dcp)
               : (tensor<2x4xf64>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C_P>) -> tensor<9x4xf64>
    call @dump_mat_dense_9x4(%7) : (tensor<9x4xf64>) -> ()

    // Release resources.
    bufferization.dealloc_tensor %sm24cc  : tensor<2x4xf64, #MAT_C_C>
    bufferization.dealloc_tensor %sm34cd  : tensor<3x4xf64, #MAT_C_D>
    bufferization.dealloc_tensor %sm44dc  : tensor<4x4xf64, #MAT_D_C>
    bufferization.dealloc_tensor %sm24ccp : tensor<2x4xf64, #MAT_C_C_P>
    bufferization.dealloc_tensor %sm34cdp : tensor<3x4xf64, #MAT_C_D_P>
    bufferization.dealloc_tensor %sm44dcp : tensor<4x4xf64, #MAT_D_C_P>
    bufferization.dealloc_tensor %4  : tensor<9x4xf64, #MAT_C_C_P>
    bufferization.dealloc_tensor %5  : tensor<9x4xf64>
    bufferization.dealloc_tensor %6  : tensor<9x4xf64, #MAT_C_C>
    bufferization.dealloc_tensor %7  : tensor<9x4xf64>
    return
  }
}
