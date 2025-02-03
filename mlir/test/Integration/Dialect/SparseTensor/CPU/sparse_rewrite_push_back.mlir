//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_libs_sve} = -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %d0 = arith.constant 0.0 : f32
    %d1 = arith.constant 1.0 : f32
    %d2 = arith.constant 2.0 : f32

    %bufferSizes = memref.alloc(%c1) : memref<?xindex>
    %buffer = memref.alloc(%c1) : memref<?xf32>

    memref.store %c0, %bufferSizes[%c0] : memref<?xindex>
    %buffer2, %s0 = sparse_tensor.push_back %c0, %buffer, %d2 : index, memref<?xf32>, f32
    %buffer3, %s1 = sparse_tensor.push_back %s0, %buffer2, %d1, %c10 : index, memref<?xf32>, f32, index

    // CHECK: 16
    %capacity = memref.dim %buffer3, %c0 : memref<?xf32>
    vector.print %capacity : index

    // CHECK: 11
    vector.print %s1 : index

    // CHECK (  2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
    %values = vector.transfer_read %buffer3[%c0], %d0: memref<?xf32>, vector<11xf32>
    vector.print %values : vector<11xf32>

    // Release the buffers.
    memref.dealloc %bufferSizes : memref<?xindex>
    memref.dealloc %buffer3 : memref<?xf32>
    return
  }
}

