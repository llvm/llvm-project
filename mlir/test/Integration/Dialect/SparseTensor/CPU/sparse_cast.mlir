// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s --sparse-compiler="vectorization-strategy=2 vl=2" | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SV = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

#trait_cast = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // A (in)
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = cast A(i)"
}

//
// Integration test that lowers a kernel annotated as sparse to actual sparse
// code, initializes a matching sparse storage scheme from a dense vector,
// and runs the resulting code with the JIT compiler.
//
module {
  //
  // Various kernels that cast a sparse vector from one type to another.
  // Arithmetic supports the following casts.
  //   sitofp
  //   uitofp
  //   fptosi
  //   fptoui
  //   extf
  //   truncf
  //   extsi
  //   extui
  //   trunci
  //   bitcast
  // Since all casts are "zero preserving" unary operations, lattice computation
  // and conversion to sparse code is straightforward.
  //
  func.func @sparse_cast_s32_to_f32(%arga: tensor<10xi32, #SV>,
                                    %argb: tensor<10xf32>) -> tensor<10xf32> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xi32, #SV>)
      outs(%argb: tensor<10xf32>) {
        ^bb(%a: i32, %x : f32):
          %cst = arith.sitofp %a : i32 to f32
          linalg.yield %cst : f32
    } -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
  func.func @sparse_cast_u32_to_f32(%arga: tensor<10xi32, #SV>,
                                    %argb: tensor<10xf32>) -> tensor<10xf32> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xi32, #SV>)
      outs(%argb: tensor<10xf32>) {
        ^bb(%a: i32, %x : f32):
          %cst = arith.uitofp %a : i32 to f32
          linalg.yield %cst : f32
    } -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
  func.func @sparse_cast_f32_to_s32(%arga: tensor<10xf32, #SV>,
                                    %argb: tensor<10xi32>) -> tensor<10xi32> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xf32, #SV>)
      outs(%argb: tensor<10xi32>) {
        ^bb(%a: f32, %x : i32):
          %cst = arith.fptosi %a : f32 to i32
          linalg.yield %cst : i32
    } -> tensor<10xi32>
    return %0 : tensor<10xi32>
  }
  func.func @sparse_cast_f64_to_u32(%arga: tensor<10xf64, #SV>,
                                    %argb: tensor<10xi32>) -> tensor<10xi32> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xf64, #SV>)
      outs(%argb: tensor<10xi32>) {
        ^bb(%a: f64, %x : i32):
          %cst = arith.fptoui %a : f64 to i32
          linalg.yield %cst : i32
    } -> tensor<10xi32>
    return %0 : tensor<10xi32>
  }
  func.func @sparse_cast_f32_to_f64(%arga: tensor<10xf32, #SV>,
                                    %argb: tensor<10xf64>) -> tensor<10xf64> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xf32, #SV>)
      outs(%argb: tensor<10xf64>) {
        ^bb(%a: f32, %x : f64):
          %cst = arith.extf %a : f32 to f64
          linalg.yield %cst : f64
    } -> tensor<10xf64>
    return %0 : tensor<10xf64>
  }
  func.func @sparse_cast_f64_to_f32(%arga: tensor<10xf64, #SV>,
                                    %argb: tensor<10xf32>) -> tensor<10xf32> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xf64, #SV>)
      outs(%argb: tensor<10xf32>) {
        ^bb(%a: f64, %x : f32):
          %cst = arith.truncf %a : f64 to f32
          linalg.yield %cst : f32
    } -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
  func.func @sparse_cast_s32_to_u64(%arga: tensor<10xi32, #SV>,
                                    %argb: tensor<10xi64>) -> tensor<10xi64> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xi32, #SV>)
      outs(%argb: tensor<10xi64>) {
        ^bb(%a: i32, %x : i64):
          %cst = arith.extsi %a : i32 to i64
          linalg.yield %cst : i64
    } -> tensor<10xi64>
    return %0 : tensor<10xi64>
  }
  func.func @sparse_cast_u32_to_s64(%arga: tensor<10xi32, #SV>,
                                    %argb: tensor<10xi64>) -> tensor<10xi64> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xi32, #SV>)
      outs(%argb: tensor<10xi64>) {
        ^bb(%a: i32, %x : i64):
          %cst = arith.extui %a : i32 to i64
          linalg.yield %cst : i64
    } -> tensor<10xi64>
    return %0 : tensor<10xi64>
  }
  func.func @sparse_cast_i32_to_i8(%arga: tensor<10xi32, #SV>,
                                   %argb: tensor<10xi8>) -> tensor<10xi8> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xi32, #SV>)
      outs(%argb: tensor<10xi8>) {
        ^bb(%a: i32, %x : i8):
          %cst = arith.trunci %a : i32 to i8
          linalg.yield %cst : i8
    } -> tensor<10xi8>
    return %0 : tensor<10xi8>
  }
  func.func @sparse_cast_f32_as_s32(%arga: tensor<10xf32, #SV>,
                                    %argb: tensor<10xi32>) -> tensor<10xi32> {
    %0 = linalg.generic #trait_cast
      ins(%arga: tensor<10xf32, #SV>)
      outs(%argb: tensor<10xi32>) {
        ^bb(%a: f32, %x : i32):
          %cst = arith.bitcast %a : f32 to i32
          linalg.yield %cst : i32
    } -> tensor<10xi32>
    return %0 : tensor<10xi32>
  }

  //
  // Main driver that converts a dense tensor into a sparse tensor
  // and then calls the sparse casting kernel.
  //
  func.func @entry() {
    %z = arith.constant 0 : index
    %b = arith.constant 0 : i8
    %i = arith.constant 0 : i32
    %l = arith.constant 0 : i64
    %f = arith.constant 0.0 : f32
    %d = arith.constant 0.0 : f64

    %zero_b = arith.constant dense<0> : tensor<10xi8>
    %zero_d = arith.constant dense<0.0> : tensor<10xf64>
    %zero_f = arith.constant dense<0.0> : tensor<10xf32>
    %zero_i = arith.constant dense<0> : tensor<10xi32>
    %zero_l = arith.constant dense<0> : tensor<10xi64>

    // Initialize dense tensors, convert to a sparse vectors.
    %0 = arith.constant dense<[ -4, -3, -2, -1, 0, 1, 2, 3, 4, 305 ]> : tensor<10xi32>
    %1 = sparse_tensor.convert %0 : tensor<10xi32> to tensor<10xi32, #SV>
    %2 = arith.constant dense<[ -4.4, -3.3, -2.2, -1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 305.5 ]> : tensor<10xf32>
    %3 = sparse_tensor.convert %2 : tensor<10xf32> to tensor<10xf32, #SV>
    %4 = arith.constant dense<[ -4.4, -3.3, -2.2, -1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 305.5 ]> : tensor<10xf64>
    %5 = sparse_tensor.convert %4 : tensor<10xf64> to tensor<10xf64, #SV>
    %6 = arith.constant dense<[ 4294967295.0, 4294967294.0, 4294967293.0, 4294967292.0,
                          0.0, 1.1, 2.2, 3.3, 4.4, 305.5 ]> : tensor<10xf64>
    %7 = sparse_tensor.convert %6 : tensor<10xf64> to tensor<10xf64, #SV>

    //
    // CHECK: ( -4, -3, -2, -1, 0, 1, 2, 3, 4, 305 )
    //
    %c0 = call @sparse_cast_s32_to_f32(%1, %zero_f) : (tensor<10xi32, #SV>, tensor<10xf32>) -> tensor<10xf32>
    %v0 = vector.transfer_read %c0[%z], %f: tensor<10xf32>, vector<10xf32>
    vector.print %v0 : vector<10xf32>

    //
    // CHECK: ( 4.29497e+09, 4.29497e+09, 4.29497e+09, 4.29497e+09, 0, 1, 2, 3, 4, 305 )
    //
    %c1 = call @sparse_cast_u32_to_f32(%1, %zero_f) : (tensor<10xi32, #SV>, tensor<10xf32>) -> tensor<10xf32>
    %v1 = vector.transfer_read %c1[%z], %f: tensor<10xf32>, vector<10xf32>
    vector.print %v1 : vector<10xf32>

    //
    // CHECK: ( -4, -3, -2, -1, 0, 1, 2, 3, 4, 305 )
    //
    %c2 = call @sparse_cast_f32_to_s32(%3, %zero_i) : (tensor<10xf32, #SV>, tensor<10xi32>) -> tensor<10xi32>
    %v2 = vector.transfer_read %c2[%z], %i: tensor<10xi32>, vector<10xi32>
    vector.print %v2 : vector<10xi32>

    //
    // CHECK: ( 4294967295, 4294967294, 4294967293, 4294967292, 0, 1, 2, 3, 4, 305 )
    //
    %c3 = call @sparse_cast_f64_to_u32(%7, %zero_i) : (tensor<10xf64, #SV>, tensor<10xi32>) -> tensor<10xi32>
    %v3 = vector.transfer_read %c3[%z], %i: tensor<10xi32>, vector<10xi32>
    %vu = vector.bitcast %v3 : vector<10xi32> to vector<10xui32>
    vector.print %vu : vector<10xui32>

    //
    // CHECK: ( -4.4, -3.3, -2.2, -1.1, 0, 1.1, 2.2, 3.3, 4.4, 305.5 )
    //
    %c4 = call @sparse_cast_f32_to_f64(%3, %zero_d) : (tensor<10xf32, #SV>, tensor<10xf64>) -> tensor<10xf64>
    %v4 = vector.transfer_read %c4[%z], %d: tensor<10xf64>, vector<10xf64>
    vector.print %v4 : vector<10xf64>

    //
    // CHECK: ( -4.4, -3.3, -2.2, -1.1, 0, 1.1, 2.2, 3.3, 4.4, 305.5 )
    //
    %c5 = call @sparse_cast_f64_to_f32(%5, %zero_f) : (tensor<10xf64, #SV>, tensor<10xf32>) -> tensor<10xf32>
    %v5 = vector.transfer_read %c5[%z], %f: tensor<10xf32>, vector<10xf32>
    vector.print %v5 : vector<10xf32>

    //
    // CHECK: ( -4, -3, -2, -1, 0, 1, 2, 3, 4, 305 )
    //
    %c6 = call @sparse_cast_s32_to_u64(%1, %zero_l) : (tensor<10xi32, #SV>, tensor<10xi64>) -> tensor<10xi64>
    %v6 = vector.transfer_read %c6[%z], %l: tensor<10xi64>, vector<10xi64>
    vector.print %v6 : vector<10xi64>

    //
    // CHECK: ( 4294967292, 4294967293, 4294967294, 4294967295, 0, 1, 2, 3, 4, 305 )
    //
    %c7 = call @sparse_cast_u32_to_s64(%1, %zero_l) : (tensor<10xi32, #SV>, tensor<10xi64>) -> tensor<10xi64>
    %v7 = vector.transfer_read %c7[%z], %l: tensor<10xi64>, vector<10xi64>
    vector.print %v7 : vector<10xi64>

    //
    // CHECK: ( -4, -3, -2, -1, 0, 1, 2, 3, 4, 49 )
    //
    %c8 = call @sparse_cast_i32_to_i8(%1, %zero_b) : (tensor<10xi32, #SV>, tensor<10xi8>) -> tensor<10xi8>
    %v8 = vector.transfer_read %c8[%z], %b: tensor<10xi8>, vector<10xi8>
    vector.print %v8 : vector<10xi8>

    //
    // CHECK: ( -1064514355, -1068289229, -1072902963, -1081291571, 0, 1066192077, 1074580685, 1079194419, 1082969293, 1134084096 )
    //
    %c9 = call @sparse_cast_f32_as_s32(%3, %zero_i) : (tensor<10xf32, #SV>, tensor<10xi32>) -> tensor<10xi32>
    %v9 = vector.transfer_read %c9[%z], %i: tensor<10xi32>, vector<10xi32>
    vector.print %v9 : vector<10xi32>

    // Release the resources.
    bufferization.dealloc_tensor %1 : tensor<10xi32, #SV>
    bufferization.dealloc_tensor %3 : tensor<10xf32, #SV>
    bufferization.dealloc_tensor %5 : tensor<10xf64, #SV>
    bufferization.dealloc_tensor %7 : tensor<10xf64, #SV>

    return
  }
}
