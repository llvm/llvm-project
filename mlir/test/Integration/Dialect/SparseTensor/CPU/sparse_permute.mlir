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
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// REDEFINE: %{env} = TENSOR0=%mlir_src_dir/test/Integration/data/mttkrp_b.tns
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | env %{env} %{run} | FileCheck %s

!Filename = !llvm.ptr

#S1 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

#S2 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d2 : compressed, d1 : compressed)
}>

#S3 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d1 : compressed, d0 : compressed, d2 : compressed)
}>

#S4 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d1 : compressed, d2 : compressed, d0 : compressed)
}>

#S5 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : compressed, d0 : compressed, d1 : compressed)
}>

#S6 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : compressed, d1 : compressed, d0 : compressed)
}>

#trait_3d = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>, // B
    affine_map<(i,j,k) -> (i,j,k)>  // A (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel"],
  doc = "A(i,j,k) = B(i,j,k)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @dump(%a: tensor<2x3x4xf64>) {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f64
    %v = vector.transfer_read %a[%c0, %c0, %c0], %f0 : tensor<2x3x4xf64>, vector<2x3x4xf64>
    vector.print %v : vector<2x3x4xf64>
    return
  }

  //// S1

  func.func @linalg1(%b: tensor<2x3x4xf64, #S1>)-> tensor<2x3x4xf64> {
    %0 = arith.constant dense<0.000000e+00> : tensor<2x3x4xf64>
    %a = linalg.generic #trait_3d
      ins(%b: tensor<2x3x4xf64, #S1>)
      outs(%0: tensor<2x3x4xf64>) {
       ^bb(%x: f64, %y: f64):
        linalg.yield %x : f64
    } -> tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @convert1(%b: tensor<2x3x4xf64, #S1>) -> tensor<2x3x4xf64> {
    %a = sparse_tensor.convert %b : tensor<2x3x4xf64, #S1> to tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @foo1(%fileName : !Filename) {
    %b = sparse_tensor.new %fileName : !Filename to tensor<2x3x4xf64, #S1>
    %0 = call @linalg1(%b) : (tensor<2x3x4xf64, #S1>) -> tensor<2x3x4xf64>
    call @dump(%0) : (tensor<2x3x4xf64>) -> ()
    %1 = call @convert1(%b) : (tensor<2x3x4xf64, #S1>) -> tensor<2x3x4xf64>
    call @dump(%1) : (tensor<2x3x4xf64>) -> ()
    bufferization.dealloc_tensor %b : tensor<2x3x4xf64, #S1>
    return
  }

  //// S2

  func.func @linalg2(%b: tensor<2x3x4xf64, #S2>)-> tensor<2x3x4xf64> {
    %0 = arith.constant dense<0.000000e+00> : tensor<2x3x4xf64>
    %a = linalg.generic #trait_3d
      ins(%b: tensor<2x3x4xf64, #S2>)
      outs(%0: tensor<2x3x4xf64>) {
       ^bb(%x: f64, %y: f64):
        linalg.yield %x : f64
    } -> tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @convert2(%b: tensor<2x3x4xf64, #S2>) -> tensor<2x3x4xf64> {
    %a = sparse_tensor.convert %b : tensor<2x3x4xf64, #S2> to tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @foo2(%fileName : !Filename) {
    %b = sparse_tensor.new %fileName : !Filename to tensor<2x3x4xf64, #S2>
    %0 = call @linalg2(%b) : (tensor<2x3x4xf64, #S2>) -> tensor<2x3x4xf64>
    call @dump(%0) : (tensor<2x3x4xf64>) -> ()
    %2 = call @convert2(%b) : (tensor<2x3x4xf64, #S2>) -> tensor<2x3x4xf64>
    call @dump(%2) : (tensor<2x3x4xf64>) -> ()
    bufferization.dealloc_tensor %b : tensor<2x3x4xf64, #S2>
    return
  }

  //// S3

  func.func @linalg3(%b: tensor<2x3x4xf64, #S3>)-> tensor<2x3x4xf64> {
    %0 = arith.constant dense<0.000000e+00> : tensor<2x3x4xf64>
    %a = linalg.generic #trait_3d
      ins(%b: tensor<2x3x4xf64, #S3>)
      outs(%0: tensor<2x3x4xf64>) {
       ^bb(%x: f64, %y: f64):
        linalg.yield %x : f64
    } -> tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @convert3(%b: tensor<2x3x4xf64, #S3>) -> tensor<2x3x4xf64> {
    %a = sparse_tensor.convert %b : tensor<2x3x4xf64, #S3> to tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @foo3(%fileName : !Filename) {
    %b = sparse_tensor.new %fileName : !Filename to tensor<2x3x4xf64, #S3>
    %0 = call @linalg3(%b) : (tensor<2x3x4xf64, #S3>) -> tensor<2x3x4xf64>
    call @dump(%0) : (tensor<2x3x4xf64>) -> ()
    %3 = call @convert3(%b) : (tensor<2x3x4xf64, #S3>) -> tensor<2x3x4xf64>
    call @dump(%3) : (tensor<2x3x4xf64>) -> ()
    bufferization.dealloc_tensor %b : tensor<2x3x4xf64, #S3>
    return
  }

  //// S4

  func.func @linalg4(%b: tensor<2x3x4xf64, #S4>)-> tensor<2x3x4xf64> {
    %0 = arith.constant dense<0.000000e+00> : tensor<2x3x4xf64>
    %a = linalg.generic #trait_3d
      ins(%b: tensor<2x3x4xf64, #S4>)
      outs(%0: tensor<2x3x4xf64>) {
       ^bb(%x: f64, %y: f64):
        linalg.yield %x : f64
    } -> tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @convert4(%b: tensor<2x3x4xf64, #S4>) -> tensor<2x3x4xf64> {
    %a = sparse_tensor.convert %b : tensor<2x3x4xf64, #S4> to tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @foo4(%fileName : !Filename) {
    %b = sparse_tensor.new %fileName : !Filename to tensor<2x3x4xf64, #S4>
    %0 = call @linalg4(%b) : (tensor<2x3x4xf64, #S4>) -> tensor<2x3x4xf64>
    call @dump(%0) : (tensor<2x3x4xf64>) -> ()
    %4 = call @convert4(%b) : (tensor<2x3x4xf64, #S4>) -> tensor<2x3x4xf64>
    call @dump(%4) : (tensor<2x3x4xf64>) -> ()
    bufferization.dealloc_tensor %b : tensor<2x3x4xf64, #S4>
    return
  }

  //// S5

  func.func @linalg5(%b: tensor<2x3x4xf64, #S5>)-> tensor<2x3x4xf64> {
    %0 = arith.constant dense<0.000000e+00> : tensor<2x3x4xf64>
    %a = linalg.generic #trait_3d
      ins(%b: tensor<2x3x4xf64, #S5>)
      outs(%0: tensor<2x3x4xf64>) {
       ^bb(%x: f64, %y: f64):
        linalg.yield %x : f64
    } -> tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @convert5(%b: tensor<2x3x4xf64, #S5>) -> tensor<2x3x4xf64> {
    %a = sparse_tensor.convert %b : tensor<2x3x4xf64, #S5> to tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @foo5(%fileName : !Filename) {
    %b = sparse_tensor.new %fileName : !Filename to tensor<2x3x4xf64, #S5>
    %0 = call @linalg5(%b) : (tensor<2x3x4xf64, #S5>) -> tensor<2x3x4xf64>
    call @dump(%0) : (tensor<2x3x4xf64>) -> ()
    %5 = call @convert5(%b) : (tensor<2x3x4xf64, #S5>) -> tensor<2x3x4xf64>
    call @dump(%5) : (tensor<2x3x4xf64>) -> ()
    bufferization.dealloc_tensor %b : tensor<2x3x4xf64, #S5>
    return
  }

  //// S6

  func.func @linalg6(%b: tensor<2x3x4xf64, #S6>)-> tensor<2x3x4xf64> {
    %0 = arith.constant dense<0.000000e+00> : tensor<2x3x4xf64>
    %a = linalg.generic #trait_3d
      ins(%b: tensor<2x3x4xf64, #S6>)
      outs(%0: tensor<2x3x4xf64>) {
       ^bb(%x: f64, %y: f64):
        linalg.yield %x : f64
    } -> tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @convert6(%b: tensor<2x3x4xf64, #S6>) -> tensor<2x3x4xf64> {
    %a = sparse_tensor.convert %b : tensor<2x3x4xf64, #S6> to tensor<2x3x4xf64>
    return %a : tensor<2x3x4xf64>
  }

  func.func @foo6(%fileName : !Filename) {
    %b = sparse_tensor.new %fileName : !Filename to tensor<2x3x4xf64, #S6>
    %0 = call @linalg6(%b) : (tensor<2x3x4xf64, #S6>) -> tensor<2x3x4xf64>
    call @dump(%0) : (tensor<2x3x4xf64>) -> ()
    %6 = call @convert6(%b) : (tensor<2x3x4xf64, #S6>) -> tensor<2x3x4xf64>
    call @dump(%6) : (tensor<2x3x4xf64>) -> ()
    bufferization.dealloc_tensor %b : tensor<2x3x4xf64, #S6>
    return
  }

  //
  // Main driver.
  //
  // CHECK-COUNT-12: ( ( ( 0, 0, 3, 63 ), ( 0, 11, 100, 0 ), ( 66, 61, 13, 43 ) ), ( ( 77, 0, 10, 46 ), ( 61, 53, 3, 75 ), ( 0, 22, 18, 0 ) ) )
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    call @foo1(%fileName) : (!Filename) -> ()
    call @foo2(%fileName) : (!Filename) -> ()
    call @foo3(%fileName) : (!Filename) -> ()
    call @foo4(%fileName) : (!Filename) -> ()
    call @foo5(%fileName) : (!Filename) -> ()
    call @foo6(%fileName) : (!Filename) -> ()
    return
  }
}
