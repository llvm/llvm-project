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

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/coordinate_zero.tns"
// RUN: %{compile} | not env %{env} %{run} 2>&1 | FileCheck %s --check-prefix=ZERO
// ZERO: Coordinate 0 is out of bounds for dimension 0 with size 4

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/coordinate_above_dimension.tns"
// RUN: %{compile} | not env %{env} %{run} 2>&1 | FileCheck %s --check-prefix=ABOVE
// ABOVE: Coordinate 5 is out of bounds for dimension 0 with size 4

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/coordinate_negative.tns"
// RUN: %{compile} | not env %{env} %{run} 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// NEGATIVE: Cannot parse coordinate for dimension 0

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/coordinate_overflow.tns"
// RUN: %{compile} | not env %{env} %{run} 2>&1 | FileCheck %s --check-prefix=OVERFLOW
// OVERFLOW: Cannot parse coordinate for dimension 0

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/coordinate_narrow.tns"
// REDEFINE: %{run_opts} = -e main_narrow -entry-point-result=void
// RUN: %{compile} | not env %{env} %{run} 2>&1 | FileCheck %s --check-prefix=NARROW
// NARROW: Coordinate 300 cannot be represented by the requested coordinate type

!Filename = !llvm.ptr

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#SparseTensor8 = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton),
  crdWidth = 8
}>

module {
  func.func private @getTensorFilename(index) -> !Filename

  func.func @main() {
    %c0 = arith.constant 0 : index
    %fileName = call @getTensorFilename(%c0) : (index) -> !Filename
    %tensor = sparse_tensor.new %fileName
      : !Filename to tensor<4x4xf64, #SparseTensor>
    sparse_tensor.print %tensor : tensor<4x4xf64, #SparseTensor>
    bufferization.dealloc_tensor %tensor
      : tensor<4x4xf64, #SparseTensor>
    return
  }

  func.func @main_narrow() {
    %c0 = arith.constant 0 : index
    %fileName = call @getTensorFilename(%c0) : (index) -> !Filename
    %tensor = sparse_tensor.new %fileName
      : !Filename to tensor<300x1xf64, #SparseTensor8>
    sparse_tensor.print %tensor : tensor<300x1xf64, #SparseTensor8>
    bufferization.dealloc_tensor %tensor
      : tensor<300x1xf64, #SparseTensor8>
    return
  }
}
