//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
// do not use these LIT config files. Hence why this is kept inline.
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

// RUN: %{compile} | %{run} | FileCheck %s
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s

#Loose = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : loose_compressed)
}>

func.func @main() {
  %pos = arith.constant dense<[1, 2, 4, 6]> : tensor<4xindex>
  %crd = arith.constant dense<[0, 1, 2, 0, 0, 3]> : tensor<6xindex>
  %val = arith.constant dense<[111.0, 10.0, 222.0, 333.0, 0.0, 20.0]> : tensor<6xf64>
  %s = sparse_tensor.assemble (%pos, %crd), %val :
    (tensor<4xindex>, tensor<6xindex>), tensor<6xf64>
    to tensor<2x4xf64, #Loose>

  %n = sparse_tensor.number_of_entries %s : tensor<2x4xf64, #Loose>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = sparse_tensor.foreach in %s init(%c0) :
    tensor<2x4xf64, #Loose>, index -> index do {
  ^bb0(%i: index, %j: index, %v: f64, %acc: index):
    %next = arith.addi %acc, %c1 : index
    sparse_tensor.yield %next : index
  }
  // The NSE is the used extent of the values storage, including holes, while
  // sparse iteration only visits the three entries stored in those intervals.
  // CHECK:      6
  // CHECK-NEXT: 3
  vector.print %n : index
  vector.print %count : index

  %has_runtime = sparse_tensor.has_runtime_library
  scf.if %has_runtime {
    // Assemble copies the buffers only on the runtime-library path.
    bufferization.dealloc_tensor %s : tensor<2x4xf64, #Loose>
  }
  return
}
