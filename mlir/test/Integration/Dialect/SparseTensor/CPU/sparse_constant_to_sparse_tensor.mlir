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
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#Tensor1  = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed"]
}>

//
// Integration tests for conversions from sparse constants to sparse tensors.
//
module {
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = arith.constant 0.0 : f64

    // A tensor in COO format.
    %ti = arith.constant sparse<[[0, 0], [0, 7], [1, 2], [4, 2], [5, 3], [6, 4], [6, 6], [9, 7]],
                          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<10x8xf64>

    // Convert the tensor in COO format to a sparse tensor with annotation #Tensor1.
    %ts = sparse_tensor.convert %ti : tensor<10x8xf64> to tensor<10x8xf64, #Tensor1>

    // CHECK: ( 0, 1, 4, 5, 6, 9 )
    %i0 = sparse_tensor.coordinates %ts { level = 0 : index } : tensor<10x8xf64, #Tensor1> to memref<?xindex>
    %i0r = vector.transfer_read %i0[%c0], %c0: memref<?xindex>, vector<6xindex>
    vector.print %i0r : vector<6xindex>

    // CHECK: ( 0, 7, 2, 2, 3, 4, 6, 7 )
    %i1 = sparse_tensor.coordinates %ts { level = 1 : index } : tensor<10x8xf64, #Tensor1> to memref<?xindex>
    %i1r = vector.transfer_read %i1[%c0], %c0: memref<?xindex>, vector<8xindex>
    vector.print %i1r : vector<8xindex>

    // CHECK: ( 1, 2, 3, 4, 5, 6, 7, 8 )
    %v = sparse_tensor.values %ts : tensor<10x8xf64, #Tensor1> to memref<?xf64>
    %vr = vector.transfer_read %v[%c0], %d0: memref<?xf64>, vector<8xf64>
    vector.print %vr : vector<8xf64>

    // Release the resources.
    bufferization.dealloc_tensor %ts : tensor<10x8xf64, #Tensor1>

    return
  }
}

