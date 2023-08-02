// DEFINE: %{option} = "enable-runtime-library=false"
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#DCSR = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ]
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
    %argx = bufferization.alloc_tensor() : tensor<4x4xi8, #DCSR>
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
    %argx = bufferization.alloc_tensor() : tensor<4x4xi8, #DCSR>
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
