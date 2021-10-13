// RUN: mlir-opt %s \
// RUN:   --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:   --sparsification --sparse-tensor-conversion \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize --lower-affine \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm \
// RUN:   --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s \
// RUN:   --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:   --sparsification="vectorization-strategy=2 vl=8" --sparse-tensor-conversion \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize --lower-affine \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm \
// RUN:   --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

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

  func @sum_reduction_i32(%arga: tensor<32xi32, #SV>,
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

  func @sum_reduction_f32(%arga: tensor<32xf32, #SV>,
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

  func @prod_reduction_i32(%arga: tensor<32xi32, #DV>,
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

  func @prod_reduction_f32(%arga: tensor<32xf32, #DV>,
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

  func @and_reduction_i32(%arga: tensor<32xi32, #DV>,
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

  func @or_reduction_i32(%arga: tensor<32xi32, #SV>,
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

  func @xor_reduction_i32(%arga: tensor<32xi32, #SV>,
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

  func @dump_i32(%arg0 : memref<i32>) {
    %v = memref.load %arg0[] : memref<i32>
    vector.print %v : i32
    return
  }

  func @dump_f32(%arg0 : memref<f32>) {
    %v = memref.load %arg0[] : memref<f32>
    vector.print %v : f32
    return
  }

  func @entry() {
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
    %m0 = memref.buffer_cast %0 : memref<i32>
    call @dump_i32(%m0) : (memref<i32>) -> ()
    %m1 = memref.buffer_cast %1 : memref<f32>
    call @dump_f32(%m1) : (memref<f32>) -> ()
    %m2 = memref.buffer_cast %2 : memref<i32>
    call @dump_i32(%m2) : (memref<i32>) -> ()
    %m3 = memref.buffer_cast %3 : memref<f32>
    call @dump_f32(%m3) : (memref<f32>) -> ()
    %m4 = memref.buffer_cast %4 : memref<i32>
    call @dump_i32(%m4) : (memref<i32>) -> ()
    %m5 = memref.buffer_cast %5 : memref<i32>
    call @dump_i32(%m5) : (memref<i32>) -> ()
    %m6 = memref.buffer_cast %6 : memref<i32>
    call @dump_i32(%m6) : (memref<i32>) -> ()

    // Release the resources.
    sparse_tensor.release %sparse_input_i32 : tensor<32xi32, #SV>
    sparse_tensor.release %sparse_input_f32 : tensor<32xf32, #SV>
    sparse_tensor.release %dense_input_i32  : tensor<32xi32, #DV>
    sparse_tensor.release %dense_input_f32  : tensor<32xf32, #DV>
    memref.dealloc %m0 : memref<i32>
    memref.dealloc %m1 : memref<f32>
    memref.dealloc %m2 : memref<i32>
    memref.dealloc %m3 : memref<f32>
    memref.dealloc %m4 : memref<i32>
    memref.dealloc %m5 : memref<i32>
    memref.dealloc %m6 : memref<i32>

    return
  }
}
