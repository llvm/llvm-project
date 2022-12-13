// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{command}

#SV = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
#DV = #sparse_tensor.encoding<{ dimLevelType = [ "dense"      ] }>

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

  func.func @prod_reduction_i32(%arga: tensor<32xi32, #DV>,
                           %argx: tensor<i32>) -> tensor<i32> {
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #DV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %x: i32):
          %0 = arith.muli %x, %a : i32
          linalg.yield %0 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  func.func @prod_reduction_f32(%arga: tensor<32xf32, #DV>,
                           %argx: tensor<f32>) -> tensor<f32> {
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xf32, #DV>)
      outs(%argx: tensor<f32>) {
        ^bb(%a: f32, %x: f32):
          %0 = arith.mulf %x, %a : f32
          linalg.yield %0 : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func @and_reduction_i32(%arga: tensor<32xi32, #DV>,
                          %argx: tensor<i32>) -> tensor<i32> {
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #DV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %x: i32):
          %0 = arith.andi %x, %a : i32
          linalg.yield %0 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
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

  func.func @entry() {
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

    // Convert constants to annotated tensors.
    %sparse_input_i32 = sparse_tensor.convert %c_0_i32
      : tensor<32xi32> to tensor<32xi32, #SV>
    %sparse_input_f32 = sparse_tensor.convert %c_0_f32
      : tensor<32xf32> to tensor<32xf32, #SV>
    %dense_input_i32 = sparse_tensor.convert %c_1_i32
      : tensor<32xi32> to tensor<32xi32, #DV>
    %dense_input_f32 = sparse_tensor.convert %c_1_f32
      : tensor<32xf32> to tensor<32xf32, #DV>

    // Call the kernels.
    %0 = call @sum_reduction_i32(%sparse_input_i32, %ri)
       : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %1 = call @sum_reduction_f32(%sparse_input_f32, %rf)
       : (tensor<32xf32, #SV>, tensor<f32>) -> tensor<f32>
    %2 = call @prod_reduction_i32(%dense_input_i32, %ri)
       : (tensor<32xi32, #DV>, tensor<i32>) -> tensor<i32>
    %3 = call @prod_reduction_f32(%dense_input_f32, %rf)
       : (tensor<32xf32, #DV>, tensor<f32>) -> tensor<f32>
    %4 = call @and_reduction_i32(%dense_input_i32, %ri)
       : (tensor<32xi32, #DV>, tensor<i32>) -> tensor<i32>
    %5 = call @or_reduction_i32(%sparse_input_i32, %ri)
       : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %6 = call @xor_reduction_i32(%sparse_input_i32, %ri)
       : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>

    // Verify results.
    //
    // CHECK: 26
    // CHECK: 27.5
    // CHECK: 3087
    // CHECK: 168
    // CHECK: 1
    // CHECK: 15
    // CHECK: 10
    //
    call @dump_i32(%0) : (tensor<i32>) -> ()
    call @dump_f32(%1) : (tensor<f32>) -> ()
    call @dump_i32(%2) : (tensor<i32>) -> ()
    call @dump_f32(%3) : (tensor<f32>) -> ()
    call @dump_i32(%4) : (tensor<i32>) -> ()
    call @dump_i32(%5) : (tensor<i32>) -> ()
    call @dump_i32(%6) : (tensor<i32>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sparse_input_i32 : tensor<32xi32, #SV>
    bufferization.dealloc_tensor %sparse_input_f32 : tensor<32xf32, #SV>
    bufferization.dealloc_tensor %dense_input_i32  : tensor<32xi32, #DV>
    bufferization.dealloc_tensor %dense_input_f32  : tensor<32xf32, #DV>

    return
  }
}
