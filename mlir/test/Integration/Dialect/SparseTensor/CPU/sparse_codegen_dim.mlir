// Test with/without runtime library, the result should always be identical.

// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=false | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=true | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

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
    return
  }
}

