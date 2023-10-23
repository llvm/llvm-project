//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparse_compiler_opts} = enable-runtime-library=true
// DEFINE: %{sparse_compiler_opts_sve} = enable-arm-sve=true %{sparse_compiler_opts}
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/ds.mtx"
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// TODO: enable!
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false
// R_UN: %{compile} | env %{env} %{run} | FileCheck %s

!Filename = !llvm.ptr<i8>

#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> ( i : dense, j : compressed)
}>

#CSR_hi = #sparse_tensor.encoding<{
  map = (i, j) -> ( i : dense, j : loose_compressed)
}>

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j ) -> ( i            : dense,
                      j floordiv 4 : dense,
                      j mod 4      : block2_4),
  crdWidth = 8
}>

module {

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Input matrix:
  //
  //  [[0.0,  0.0,  1.0,  2.0,  0.0,  3.0,  0.0,  4.0],
  //   [0.0,  5.0,  6.0,  0.0,  7.0,  0.0,  0.0,  8.0],
  //   [9.0,  0.0, 10.0,  0.0, 11.0, 12.0,  0.0,  0.0]]
  //
  func.func @entry() {
    %u0 = arith.constant 0   : i8
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %A1 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #CSR>
    %A2 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #CSR_hi>
    %A3 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #NV_24>

    //
    // CSR:
    //
    // CHECK:      ( 0, 4, 8, 12 )
    // CHECK-NEXT: ( 2, 3, 5, 7, 1, 2, 4, 7, 0, 2, 4, 5 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
    //
    %pos1 = sparse_tensor.positions %A1 {level = 1 : index } : tensor<?x?xf64, #CSR> to memref<?xindex>
    %vecp1 = vector.transfer_read %pos1[%c0], %c0 : memref<?xindex>, vector<4xindex>
    vector.print %vecp1 : vector<4xindex>
    %crd1 = sparse_tensor.coordinates %A1 {level = 1 : index } : tensor<?x?xf64, #CSR> to memref<?xindex>
    %vecc1 = vector.transfer_read %crd1[%c0], %c0 : memref<?xindex>, vector<12xindex>
    vector.print %vecc1 : vector<12xindex>
    %val1 = sparse_tensor.values %A1 : tensor<?x?xf64, #CSR> to memref<?xf64>
    %vecv1 = vector.transfer_read %val1[%c0], %f0 : memref<?xf64>, vector<12xf64>
    vector.print %vecv1 : vector<12xf64>

    //
    // CSR_hi:
    //
    // CHECK-NEXT: ( 0, 4, 4, 8, 8, 12 )
    // CHECK-NEXT: ( 2, 3, 5, 7, 1, 2, 4, 7, 0, 2, 4, 5 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
    //
    %pos2 = sparse_tensor.positions %A2 {level = 1 : index } : tensor<?x?xf64, #CSR_hi> to memref<?xindex>
    %vecp2 = vector.transfer_read %pos2[%c0], %c0 : memref<?xindex>, vector<6xindex>
    vector.print %vecp2 : vector<6xindex>
    %crd2 = sparse_tensor.coordinates %A2 {level = 1 : index } : tensor<?x?xf64, #CSR_hi> to memref<?xindex>
    %vecc2 = vector.transfer_read %crd2[%c0], %c0 : memref<?xindex>, vector<12xindex>
    vector.print %vecc2 : vector<12xindex>
    %val2 = sparse_tensor.values %A2 : tensor<?x?xf64, #CSR_hi> to memref<?xf64>
    %vecv2 = vector.transfer_read %val2[%c0], %f0 : memref<?xf64>, vector<12xf64>
    vector.print %vecv2 : vector<12xf64>

    //
    // NV_24
    //
    // CHECK-NEXT: ( 2, 3, 1, 3, 1, 2, 0, 3, 0, 2, 0, 1 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
    //
    %crd3 = sparse_tensor.coordinates %A3 {level = 2 : index } : tensor<?x?xf64, #NV_24> to memref<?xi8>
    %vecc3 = vector.transfer_read %crd3[%c0], %u0 : memref<?xi8>, vector<12xi8>
    vector.print %vecc3 : vector<12xi8>
    %val3 = sparse_tensor.values %A3 : tensor<?x?xf64, #NV_24> to memref<?xf64>
    %vecv3 = vector.transfer_read %val3[%c0], %f0 : memref<?xf64>, vector<12xf64>
    vector.print %vecv3 : vector<12xf64>

    // Release the resources.
    bufferization.dealloc_tensor %A1: tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %A2: tensor<?x?xf64, #CSR_hi>
    bufferization.dealloc_tensor %A3: tensor<?x?xf64, #NV_24>

    return
  }
}
