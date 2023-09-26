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
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
#DV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : dense) }>

#trait_reduction = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> ()>    // x (scalar out)
  ],
  iterator_types = ["reduction"],
  doc = "x += PROD_CUSTOM_i a(i)"
}

// An example of vector reductions.
module {

  // Custom prod reduction: stored i32 elements only.
  func.func @prod_dreduction_i32(%arga: tensor<32xi32, #DV>,
                                 %argx: tensor<i32>) -> tensor<i32> {
    %c = tensor.extract %argx[] : tensor<i32>
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #DV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
          %1 = sparse_tensor.reduce %a, %b, %c : i32 {
            ^bb0(%x: i32, %y: i32):
              %2 = arith.muli %x, %y : i32
              sparse_tensor.yield %2 : i32
          }
          linalg.yield %1 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  // Custom prod reduction: stored f32 elements only.
  func.func @prod_dreduction_f32(%arga: tensor<32xf32, #DV>,
                                 %argx: tensor<f32>) -> tensor<f32> {
    %c = tensor.extract %argx[] : tensor<f32>
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xf32, #DV>)
      outs(%argx: tensor<f32>) {
        ^bb(%a: f32, %b: f32):
          %1 = sparse_tensor.reduce %a, %b, %c : f32 {
            ^bb0(%x: f32, %y: f32):
              %2 = arith.mulf %x, %y : f32
              sparse_tensor.yield %2 : f32
          }
          linalg.yield %1 : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
  }

  // Custom prod reduction: stored i32 elements only.
  func.func @prod_sreduction_i32(%arga: tensor<32xi32, #SV>,
                                 %argx: tensor<i32>) -> tensor<i32> {
    %c = tensor.extract %argx[] : tensor<i32>
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
          %1 = sparse_tensor.reduce %a, %b, %c : i32 {
            ^bb0(%x: i32, %y: i32):
              %2 = arith.muli %x, %y : i32
              sparse_tensor.yield %2 : i32
          }
          linalg.yield %1 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  // Custom prod reduction: stored f32 elements only.
  func.func @prod_sreduction_f32(%arga: tensor<32xf32, #SV>,
                                 %argx: tensor<f32>) -> tensor<f32> {
    %c = tensor.extract %argx[] : tensor<f32>
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xf32, #SV>)
      outs(%argx: tensor<f32>) {
        ^bb(%a: f32, %b: f32):
          %1 = sparse_tensor.reduce %a, %b, %c : f32 {
            ^bb0(%x: f32, %y: f32):
              %2 = arith.mulf %x, %y : f32
              sparse_tensor.yield %2 : f32
          }
          linalg.yield %1 : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
  }

  // Custom prod reduction: stored i32 elements and implicit zeros.
  //
  // NOTE: this is a somewhat strange operation, since for most sparse
  //       situations the outcome would always be zero; it is added
  //       to test full functionality and illustrate the subtle differences
  //       between the various custom operations; it would make a bit more
  //       sense for e.g. a min/max reductions, although it still would
  //       "densify" the iteration space.
  //
  func.func @prod_xreduction_i32(%arga: tensor<32xi32, #SV>,
                                 %argx: tensor<i32>) -> tensor<i32> {
    %c = tensor.extract %argx[] : tensor<i32>
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
           %u = sparse_tensor.unary %a : i32 to i32
           present={
             ^bb0(%x: i32):
             sparse_tensor.yield %x : i32
           } absent={
             ^bb0:
             %c0 = arith.constant 0 : i32
             sparse_tensor.yield %c0 : i32
          }
          %1 = sparse_tensor.reduce %u, %b, %c : i32 {
            ^bb0(%x: i32, %y: i32):
              %2 = arith.muli %x, %y : i32
              sparse_tensor.yield %2 : i32
          }
          linalg.yield %1 : i32
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

  func.func @entry() {
    %ri = arith.constant dense< 7   > : tensor<i32>
    %rf = arith.constant dense< 2.0 > : tensor<f32>

    // Vectors with a few zeros.
    %c_0_i32 = arith.constant dense<[
      1, 1, 7, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 3, 0, 1, 1, 1, 1, 1, 0, 1, 1, 7, 3
    ]> : tensor<32xi32>

    %c_0_f32 = arith.constant dense<[
      1.0, 1.0, 1.0, 3.5, 1.0, 1.0, 1.0, 1.0,
      1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
      1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
    ]> : tensor<32xf32>

    // Vectors with no zeros.
    %c_1_i32 = arith.constant dense<[
      1, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 3
    ]> : tensor<32xi32>

    %c_1_f32 = arith.constant dense<[
      1.0, 1.0, 1.0, 3.5, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0
    ]> : tensor<32xf32>

    // Convert constants to annotated tensors. Note that this
    // particular conversion only stores nonzero elements,
    // so we will have no explicit zeros, only implicit zeros.
    %d0_i32 = sparse_tensor.convert %c_0_i32
      : tensor<32xi32> to tensor<32xi32, #DV>
    %d0_f32 = sparse_tensor.convert %c_0_f32
      : tensor<32xf32> to tensor<32xf32, #DV>
    %s0_i32 = sparse_tensor.convert %c_0_i32
      : tensor<32xi32> to tensor<32xi32, #SV>
    %s0_f32 = sparse_tensor.convert %c_0_f32
      : tensor<32xf32> to tensor<32xf32, #SV>
    %d1_i32 = sparse_tensor.convert %c_1_i32
      : tensor<32xi32> to tensor<32xi32, #DV>
    %d1_f32 = sparse_tensor.convert %c_1_f32
      : tensor<32xf32> to tensor<32xf32, #DV>
    %s1_i32 = sparse_tensor.convert %c_1_i32
      : tensor<32xi32> to tensor<32xi32, #SV>
    %s1_f32 = sparse_tensor.convert %c_1_f32
      : tensor<32xf32> to tensor<32xf32, #SV>

    // Special case, construct a sparse vector with an explicit zero.
    %v0 = arith.constant sparse< [ [1] ], [ 0 ] > : tensor<32xi32>
    %s0 = sparse_tensor.convert %v0: tensor<32xi32> to tensor<32xi32, #SV>

    // Call the kernels.
    %0 = call @prod_dreduction_i32(%d0_i32, %ri) : (tensor<32xi32, #DV>, tensor<i32>) -> tensor<i32>
    %1 = call @prod_dreduction_f32(%d0_f32, %rf) : (tensor<32xf32, #DV>, tensor<f32>) -> tensor<f32>
    %2 = call @prod_sreduction_i32(%s0_i32, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %3 = call @prod_sreduction_f32(%s0_f32, %rf) : (tensor<32xf32, #SV>, tensor<f32>) -> tensor<f32>
    %4 = call @prod_dreduction_i32(%d1_i32, %ri) : (tensor<32xi32, #DV>, tensor<i32>) -> tensor<i32>
    %5 = call @prod_dreduction_f32(%d1_f32, %rf) : (tensor<32xf32, #DV>, tensor<f32>) -> tensor<f32>
    %6 = call @prod_sreduction_i32(%s1_i32, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %7 = call @prod_sreduction_f32(%s1_f32, %rf) : (tensor<32xf32, #SV>, tensor<f32>) -> tensor<f32>
    %8 = call @prod_sreduction_i32(%s0,     %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %9 = call @prod_xreduction_i32(%s0_i32, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %10 = call @prod_xreduction_i32(%s1_i32, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>

    // Verify results. Note that the custom reduction gave permission
    // to treat an explicit vs implicit zero differently to compute the
    // full product reduction over stored elements. A "standard" product
    // reduction would have to return 0 for any implicit zero occurrence
    // too. An explicit zero nullifies the product, though, as requested.
    //
    // CHECK: 0
    // CHECK: 0
    // CHECK: 3087
    // CHECK: 14
    // CHECK: 3087
    // CHECK: 168
    // CHECK: 3087
    // CHECK: 168
    // CHECK: 0
    // CHECK: 0
    // CHECK: 3087
    //
    call @dump_i32(%0) : (tensor<i32>) -> ()
    call @dump_f32(%1) : (tensor<f32>) -> ()
    call @dump_i32(%2) : (tensor<i32>) -> ()
    call @dump_f32(%3) : (tensor<f32>) -> ()
    call @dump_i32(%4) : (tensor<i32>) -> ()
    call @dump_f32(%5) : (tensor<f32>) -> ()
    call @dump_i32(%6) : (tensor<i32>) -> ()
    call @dump_f32(%7) : (tensor<f32>) -> ()
    call @dump_i32(%8) : (tensor<i32>) -> ()
    call @dump_i32(%9) : (tensor<i32>) -> ()
    call @dump_i32(%10) : (tensor<i32>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %d0_i32 : tensor<32xi32, #DV>
    bufferization.dealloc_tensor %d0_f32 : tensor<32xf32, #DV>
    bufferization.dealloc_tensor %s0_i32 : tensor<32xi32, #SV>
    bufferization.dealloc_tensor %s0_f32 : tensor<32xf32, #SV>
    bufferization.dealloc_tensor %d1_i32 : tensor<32xi32, #DV>
    bufferization.dealloc_tensor %d1_f32 : tensor<32xf32, #DV>
    bufferization.dealloc_tensor %s1_i32 : tensor<32xi32, #SV>
    bufferization.dealloc_tensor %s1_f32 : tensor<32xf32, #SV>
    bufferization.dealloc_tensor %s0     : tensor<32xi32, #SV>

    return
  }
}
