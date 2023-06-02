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
  // Tests without permutation.
  //

  // Concats all sparse matrices (with different encodings) to a sparse matrix.
  func.func @concat_sparse_sparse(%arg0: tensor<2x4xf64, #MAT_C_C>, %arg1: tensor<3x4xf64, #MAT_C_D>, %arg2: tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64, #MAT_C_C> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #MAT_C_C>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C> to tensor<9x4xf64, #MAT_C_C>
    return %0 : tensor<9x4xf64, #MAT_C_C>
  }

  // Concats all sparse matrices (with different encodings) to a dense matrix.
  func.func @concat_sparse_dense(%arg0: tensor<2x4xf64, #MAT_C_C>, %arg1: tensor<3x4xf64, #MAT_C_D>, %arg2: tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #MAT_C_C>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C> to tensor<9x4xf64>
    return %0 : tensor<9x4xf64>
  }

  // Concats mix sparse and dense matrices to a sparse matrix.
  func.func @concat_mix_sparse(%arg0: tensor<2x4xf64>, %arg1: tensor<3x4xf64, #MAT_C_D>, %arg2: tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64, #MAT_C_C> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C> to tensor<9x4xf64, #MAT_C_C>
    return %0 : tensor<9x4xf64, #MAT_C_C>
  }

  // Concats mix sparse and dense matrices to a dense matrix.
  func.func @concat_mix_dense(%arg0: tensor<2x4xf64>, %arg1: tensor<3x4xf64, #MAT_C_D>, %arg2: tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C> to tensor<9x4xf64>
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

  func.func @dump_mat_dense_9x4(%A: tensor<9x4xf64>) {
    %u = tensor.cast %A : tensor<9x4xf64> to tensor<*xf64>
    call @printMemrefF64(%u) : (tensor<*xf64>) -> ()

    return
  }

  // Driver method to call and verify kernels.
  func.func @entry() {
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
    %0 = call @concat_sparse_sparse(%sm24cc, %sm34cd, %sm44dc)
               : (tensor<2x4xf64, #MAT_C_C>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64, #MAT_C_C>
    call @dump_mat_9x4(%0) : (tensor<9x4xf64, #MAT_C_C>) -> ()

    // CHECK:      {{\[}}[1,   0,   3,   0],
    // CHECK-NEXT:  [0,   2,   0,   0],
    // CHECK-NEXT:  [1,   0,   1,   1],
    // CHECK-NEXT:  [0,   0.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [0,   0,   1.5,   1],
    // CHECK-NEXT:  [0,   3.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [1,   0.5,   0,   0]]
    %1 = call @concat_sparse_dense(%sm24cc, %sm34cd, %sm44dc)
               : (tensor<2x4xf64, #MAT_C_C>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64>
    call @dump_mat_dense_9x4(%1) : (tensor<9x4xf64>) -> ()

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
    %2 = call @concat_mix_sparse(%m24, %sm34cd, %sm44dc)
               : (tensor<2x4xf64>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64, #MAT_C_C>
    call @dump_mat_9x4(%2) : (tensor<9x4xf64, #MAT_C_C>) -> ()

    // CHECK:      {{\[}}[1,   0,   3,   0],
    // CHECK-NEXT:  [0,   2,   0,   0],
    // CHECK-NEXT:  [1,   0,   1,   1],
    // CHECK-NEXT:  [0,   0.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [0,   0,   1.5,   1],
    // CHECK-NEXT:  [0,   3.5,   0,   0],
    // CHECK-NEXT:  [1,   5,   2,   0],
    // CHECK-NEXT:  [1,   0.5,   0,   0]]
    %3 = call @concat_mix_dense(%m24, %sm34cd, %sm44dc)
               : (tensor<2x4xf64>, tensor<3x4xf64, #MAT_C_D>, tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64>
    call @dump_mat_dense_9x4(%3) : (tensor<9x4xf64>) -> ()


    // Release resources.
    bufferization.dealloc_tensor %sm24cc  : tensor<2x4xf64, #MAT_C_C>
    bufferization.dealloc_tensor %sm34cd  : tensor<3x4xf64, #MAT_C_D>
    bufferization.dealloc_tensor %sm44dc  : tensor<4x4xf64, #MAT_D_C>
    bufferization.dealloc_tensor %0  : tensor<9x4xf64, #MAT_C_C>
    bufferization.dealloc_tensor %1  : tensor<9x4xf64>
    bufferization.dealloc_tensor %2  : tensor<9x4xf64, #MAT_C_C>
    bufferization.dealloc_tensor %3  : tensor<9x4xf64>
    return
  }
}
