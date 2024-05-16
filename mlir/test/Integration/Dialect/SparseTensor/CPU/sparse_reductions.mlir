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

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

// Reduction in this file _are_ supported by the AArch64 SVE backend

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

#trait_reduction = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> ()>    // x (scalar out)
  ],
  iterator_types = ["reduction"],
  doc = "x += OPER_i a(i)"
}

// An example of vector reductions.
module {

  func.func @sum_reduction_i32(%arga: tensor<32xi32, #SV>,
                          %argx: tensor<i32>) -> tensor<i32> {
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %x: i32):
          %0 = arith.addi %x, %a : i32
          linalg.yield %0 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  func.func @sum_reduction_f32(%arga: tensor<32xf32, #SV>,
                          %argx: tensor<f32>) -> tensor<f32> {
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xf32, #SV>)
      outs(%argx: tensor<f32>) {
        ^bb(%a: f32, %x: f32):
          %0 = arith.addf %x, %a : f32
          linalg.yield %0 : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func @or_reduction_i32(%arga: tensor<32xi32, #SV>,
                         %argx: tensor<i32>) -> tensor<i32> {
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %x: i32):
          %0 = arith.ori %x, %a : i32
          linalg.yield %0 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  func.func @xor_reduction_i32(%arga: tensor<32xi32, #SV>,
                          %argx: tensor<i32>) -> tensor<i32> {
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %x: i32):
          %0 = arith.xori %x, %a : i32
          linalg.yield %0 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  func.func @dump_i32(%arg0 : tensor<i32>) {
    %v = tensor.extract %arg0[] : tensor<i32>
    vector.print %v : i32
    return
  }

  func.func @dump_f32(%arg0 : tensor<f32>) {
    %v = tensor.extract %arg0[] : tensor<f32>
    vector.print %v : f32
    return
  }

  func.func @main() {
    %ri = arith.constant dense< 7   > : tensor<i32>
    %rf = arith.constant dense< 2.0 > : tensor<f32>

    %c_0_i32 = arith.constant dense<[
      0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0,
      0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0
    ]> : tensor<32xi32>

    %c_0_f32 = arith.constant dense<[
      0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0,
      2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 9.0
    ]> : tensor<32xf32>

    // Convert constants to annotated tensors.
    %sparse_input_i32 = sparse_tensor.convert %c_0_i32
      : tensor<32xi32> to tensor<32xi32, #SV>
    %sparse_input_f32 = sparse_tensor.convert %c_0_f32
      : tensor<32xf32> to tensor<32xf32, #SV>

    // Call the kernels.
    %0 = call @sum_reduction_i32(%sparse_input_i32, %ri)
       : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %1 = call @sum_reduction_f32(%sparse_input_f32, %rf)
       : (tensor<32xf32, #SV>, tensor<f32>) -> tensor<f32>
    %2 = call @or_reduction_i32(%sparse_input_i32, %ri)
       : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %3 = call @xor_reduction_i32(%sparse_input_i32, %ri)
       : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>

    // Verify results.
    //
    // CHECK: 26
    // CHECK: 27.5
    // CHECK: 15
    // CHECK: 10
    //
    call @dump_i32(%0) : (tensor<i32>) -> ()
    call @dump_f32(%1) : (tensor<f32>) -> ()
    call @dump_i32(%2) : (tensor<i32>) -> ()
    call @dump_i32(%3) : (tensor<i32>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sparse_input_i32 : tensor<32xi32, #SV>
    bufferization.dealloc_tensor %sparse_input_f32 : tensor<32xf32, #SV>
    bufferization.dealloc_tensor %0 : tensor<i32>
    bufferization.dealloc_tensor %1 : tensor<f32>
    bufferization.dealloc_tensor %2 : tensor<i32>
    bufferization.dealloc_tensor %3 : tensor<i32>

    return
  }
}
