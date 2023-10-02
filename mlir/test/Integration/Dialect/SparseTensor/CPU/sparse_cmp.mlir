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

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (i,j)>, // B
    affine_map<(i,j) -> (i,j)>  // x (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i, j) = cmp A(i,j) B(i, j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  func.func @cmp_all_dense(%arga: tensor<4x4xf64>,
                           %argb: tensor<4x4xf64>,
                           %argx: tensor<4x4xi8>) -> tensor<4x4xi8> {
    %0 = linalg.generic #trait
       ins(%arga, %argb: tensor<4x4xf64>, tensor<4x4xf64>)
      outs(%argx: tensor<4x4xi8>) {
        ^bb(%a: f64, %b: f64, %x: i8):
          %0 = arith.cmpf ult, %a, %b : f64
          %1 = arith.extui %0 : i1 to i8
          linalg.yield %1 : i8
    } -> tensor<4x4xi8>
    return %0 : tensor<4x4xi8>
  }

  func.func @cmp_lhs_sparse(%arga: tensor<4x4xf64, #DCSR>,
                            %argb: tensor<4x4xf64>) -> tensor<4x4xi8, #DCSR> {
    %argx = tensor.empty() : tensor<4x4xi8, #DCSR>
    %0 = linalg.generic #trait
       ins(%arga, %argb: tensor<4x4xf64, #DCSR>, tensor<4x4xf64>)
      outs(%argx: tensor<4x4xi8, #DCSR>) {
        ^bb(%a: f64, %b: f64, %x: i8):
          %0 = arith.cmpf ult, %a, %b : f64
          %1 = arith.extui %0 : i1 to i8
          linalg.yield %1 : i8
    } -> tensor<4x4xi8, #DCSR>
    return %0 : tensor<4x4xi8, #DCSR>
  }

  func.func @cmp_all_sparse(%arga: tensor<4x4xf64, #DCSR>,
                            %argb: tensor<4x4xf64, #DCSR>) -> tensor<4x4xi8, #DCSR> {
    %argx = tensor.empty() : tensor<4x4xi8, #DCSR>
    %0 = linalg.generic #trait
       ins(%arga, %argb: tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>)
      outs(%argx: tensor<4x4xi8, #DCSR>) {
        ^bb(%a: f64, %b: f64, %x: i8):
          %0 = arith.cmpf ult, %a, %b : f64
          %1 = arith.extui %0 : i1 to i8
          linalg.yield %1 : i8
    } -> tensor<4x4xi8, #DCSR>
    return %0 : tensor<4x4xi8, #DCSR>
  }

  //
  // Main driver that constructs matrix and calls the sparse kernel to perform
  // element-wise comparison.
  //
  func.func @entry() {
    %d0 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index

    %lhs_dn = arith.constant dense<
      [ [ 0.0, 0.0, 1.5, 1.0],
        [ 0.0, 3.5, 0.0, 0.0],
        [ 1.0, 5.0, 2.0, 0.0],
        [ 1.0, 0.5, 0.0, 0.0] ]> : tensor<4x4xf64>

    %rhs_dn = arith.constant dense<
      [ [ 0.0, 1.5, 1.0, 1.5],
        [ 3.5, 0.0, 0.0, 0.0],
        [ 5.0, 2.0, 0.0, 2.0],
        [ 0.5, 0.0, 0.0, 0.0] ]> : tensor<4x4xf64>

    %lhs_sp = sparse_tensor.convert %lhs_dn : tensor<4x4xf64> to tensor<4x4xf64, #DCSR>
    %rhs_sp = sparse_tensor.convert %rhs_dn : tensor<4x4xf64> to tensor<4x4xf64, #DCSR>

    %output = arith.constant dense<0> : tensor<4x4xi8>
    %all_dn_out = call @cmp_all_dense(%lhs_dn, %rhs_dn, %output)
            : (tensor<4x4xf64>, tensor<4x4xf64>, tensor<4x4xi8>) -> tensor<4x4xi8>
    %lhs_sp_out = call @cmp_lhs_sparse(%lhs_sp, %rhs_dn)
            : (tensor<4x4xf64, #DCSR>, tensor<4x4xf64>) -> tensor<4x4xi8, #DCSR>
    %all_sp_out = call @cmp_all_sparse(%lhs_sp, %rhs_sp)
            : (tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>) -> tensor<4x4xi8, #DCSR>

    //
    // All should have the same result.
    //
    // CHECK-COUNT-3: ( ( 0, 1, 0, 1 ), ( 1, 0, 0, 0 ), ( 1, 0, 0, 1 ), ( 0, 0, 0, 0 ) )
    %v = vector.transfer_read %all_dn_out[%c0, %c0], %d0
       : tensor<4x4xi8>, vector<4x4xi8>
    vector.print %v : vector<4x4xi8>

    %lhs_sp_ret = sparse_tensor.convert %lhs_sp_out
      : tensor<4x4xi8, #DCSR> to tensor<4x4xi8>
    %v1 = vector.transfer_read %lhs_sp_ret[%c0, %c0], %d0
      : tensor<4x4xi8>, vector<4x4xi8>
    vector.print %v1 : vector<4x4xi8>

    %rhs_sp_ret = sparse_tensor.convert %all_sp_out
      : tensor<4x4xi8, #DCSR> to tensor<4x4xi8>
    %v2 = vector.transfer_read %rhs_sp_ret[%c0, %c0], %d0
      : tensor<4x4xi8>, vector<4x4xi8>
    vector.print %v2 : vector<4x4xi8>


    bufferization.dealloc_tensor %lhs_sp : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %rhs_sp : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %lhs_sp_out : tensor<4x4xi8, #DCSR>
    bufferization.dealloc_tensor %all_sp_out : tensor<4x4xi8, #DCSR>

    return
  }
}
