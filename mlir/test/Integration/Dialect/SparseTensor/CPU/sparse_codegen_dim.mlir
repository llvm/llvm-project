// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#DCSR  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed"]
}>

module {
  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %t1 = bufferization.alloc_tensor() : tensor<4x5xf64, #DCSR>
    %t2 = bufferization.alloc_tensor(%c2, %c3) : tensor<?x?xf64, #DCSR>

    %d1_0 = tensor.dim %t1, %c0 : tensor<4x5xf64, #DCSR>
    %d2_0 = tensor.dim %t2, %c0 : tensor<?x?xf64, #DCSR>
    %d1_1 = tensor.dim %t1, %c1 : tensor<4x5xf64, #DCSR>
    %d2_1 = tensor.dim %t2, %c1 : tensor<?x?xf64, #DCSR>

    // CHECK: 4
    vector.print %d1_0 : index
    // CHECK-NEXT: 2
    vector.print %d2_0 : index
    // CHECK-NEXT: 5
    vector.print %d1_1 : index
    // CHECK-NEXT: 3
    vector.print %d2_1 : index

    // Release resources.
    bufferization.dealloc_tensor %t1 : tensor<4x5xf64, #DCSR>
    bufferization.dealloc_tensor %t2 : tensor<?x?xf64, #DCSR>

    return
  }
}

