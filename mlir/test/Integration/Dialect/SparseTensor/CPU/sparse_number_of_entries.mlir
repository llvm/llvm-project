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
#LooseDense = #sparse_tensor.encoding<{
  map = (i, j, k) -> (i : dense, j : loose_compressed, k : dense),
  posWidth = 32,
  crdWidth = 32
}>
#LooseCompressed = #sparse_tensor.encoding<{
  map = (i, j, k) -> (i : dense, j : loose_compressed, k : compressed)
}>

func.func @check(%pos: tensor<4xindex>, %crd: tensor<6xindex>,
                 %val: tensor<6xf64>) {
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
  vector.print %n : index
  vector.print %count : index
  %has_runtime = sparse_tensor.has_runtime_library
  scf.if %has_runtime {
    // Assemble copies the buffers only on the runtime-library path.
    bufferization.dealloc_tensor %s : tensor<2x4xf64, #Loose>
  }
  return
}

func.func @main() {
  %crd = arith.constant dense<[0, 1, 2, 0, 0, 3]> : tensor<6xindex>
  %val = arith.constant dense<[0.0, 10.0, 20.0, 999.0, 0.0, 20.0]> : tensor<6xf64>
  %contiguous = arith.constant dense<[0, 1, 1, 3]> : tensor<4xindex>
  %gaps = arith.constant dense<[1, 2, 4, 6]> : tensor<4xindex>
  %empty = arith.constant dense<[1, 1, 6, 6]> : tensor<4xindex>

  // Contiguous intervals ignore unused trailing capacity.
  // CHECK:      3
  // CHECK-NEXT: 3
  call @check(%contiguous, %crd, %val) : (tensor<4xindex>, tensor<6xindex>, tensor<6xf64>) -> ()
  // Ignore leading/interior holes, but include the explicit zero at slot 4.
  // CHECK-NEXT: 3
  // CHECK-NEXT: 3
  call @check(%gaps, %crd, %val) : (tensor<4xindex>, tensor<6xindex>, tensor<6xf64>) -> ()
  // Nonzero positions do not imply any stored entries.
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  call @check(%empty, %crd, %val) : (tensor<4xindex>, tensor<6xindex>, tensor<6xf64>) -> ()

  // A non-innermost loose level followed by a dense level. Every stored
  // coordinate has two values, including explicitly stored zeros.
  %pos32 = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi32>
  %crd32 = arith.constant dense<[0, 1, 2, 0, 0, 3]> : tensor<6xi32>
  %dense_values = arith.constant dense<0.0> : tensor<12xf64>
  %dense = sparse_tensor.assemble (%pos32, %crd32), %dense_values :
    (tensor<4xi32>, tensor<6xi32>), tensor<12xf64>
    to tensor<2x4x2xf64, #LooseDense>
  %dense_n = sparse_tensor.number_of_entries %dense : tensor<2x4x2xf64, #LooseDense>
  // CHECK-NEXT: 6
  vector.print %dense_n : index

  // Only the compressed children of reachable loose coordinates count.
  // Slots 1, 4 and 5 have respectively 2, 2 and 1 children. Summing the
  // lengths of all compressed intervals would incorrectly count 8 entries.
  %child_pos = arith.constant dense<[0, 1, 3, 4, 5, 7, 8]> : tensor<7xindex>
  %child_crd = arith.constant dense<[0, 0, 1, 0, 1, 0, 1, 1]> : tensor<8xindex>
  %child_val = arith.constant dense<0.0> : tensor<8xf64>
  %compressed = sparse_tensor.assemble (%gaps, %crd, %child_pos, %child_crd), %child_val :
    (tensor<4xindex>, tensor<6xindex>, tensor<7xindex>, tensor<8xindex>), tensor<8xf64>
    to tensor<2x4x2xf64, #LooseCompressed>
  %compressed_n = sparse_tensor.number_of_entries %compressed : tensor<2x4x2xf64, #LooseCompressed>
  // CHECK-NEXT: 5
  vector.print %compressed_n : index

  %has_runtime = sparse_tensor.has_runtime_library
  scf.if %has_runtime {
    bufferization.dealloc_tensor %dense : tensor<2x4x2xf64, #LooseDense>
    bufferization.dealloc_tensor %compressed : tensor<2x4x2xf64, #LooseCompressed>
  }
  return
}
