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

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/block.mtx"
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// TODO: enable!
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false
// R_UN: %{compile} | env %{env} %{run} | FileCheck %s

#BSR = #sparse_tensor.encoding<{
  map = (i, j) ->
    ( i floordiv 2 : dense
    , j floordiv 2 : compressed
    , i mod 2 : dense
    , j mod 2 : dense
    )
}>

!Filename = !llvm.ptr<i8>

//
// Example 2x2 block storage:
//
//  +-----+-----+-----+    +-----+-----+-----+
//  | 1 2 | . . | 4 . |    | 1 2 |     | 4 0 |
//  | . 3 | . . | . 5 |    | 0 3 |     | 0 5 |
//  +-----+-----+-----+ => +-----+-----+-----+
//  | . . | 6 7 | . . |    |     | 6 7 |     |
//  | . . | 8 . | . . |    |     | 8 0 |     |
//  +-----+-----+-----+    +-----+-----+-----+
//
// Stored as:
//
//    positions[1]   : 0 2 3
//    coordinates[1] : 0 2 1
//    values         : 1.000000 2.000000 0.000000 3.000000 4.000000 0.000000 0.000000 5.000000 6.000000 7.000000 8.000000 0.000000
//
module {

  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @entry() {
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %A = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #BSR>

    // CHECK:      ( 0, 2, 3 )
    // CHECK-NEXT: ( 0, 2, 1 )
    // CHECK-NEXT: ( 1, 2, 0, 3, 4, 0, 0, 5, 6, 7, 8, 0 )
    %pos = sparse_tensor.positions %A {level = 1 : index } : tensor<?x?xf64, #BSR> to memref<?xindex>
    %vecp = vector.transfer_read %pos[%c0], %c0 : memref<?xindex>, vector<3xindex>
    vector.print %vecp : vector<3xindex>
    %crd = sparse_tensor.coordinates %A {level = 1 : index } : tensor<?x?xf64, #BSR> to memref<?xindex>
    %vecc = vector.transfer_read %crd[%c0], %c0 : memref<?xindex>, vector<3xindex>
    vector.print %vecc : vector<3xindex>
    %val = sparse_tensor.values %A : tensor<?x?xf64, #BSR> to memref<?xf64>
    %vecv = vector.transfer_read %val[%c0], %f0 : memref<?xf64>, vector<12xf64>
    vector.print %vecv : vector<12xf64>

    // Release the resources.
    bufferization.dealloc_tensor %A: tensor<?x?xf64, #BSR>

    return
  }
}
