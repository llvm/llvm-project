// RUN: mlir-opt %s \
// RUN: -func-bufferize -one-shot-bufferize="bufferize-function-boundaries" --canonicalize \
// RUN:   -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -finalize-memref-to-llvm\
// RUN:   -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils |\
// RUN: FileCheck %s

module {
  func.func private @verifyMemRefI8(%a : tensor<*xi8>, %b : tensor<*xi8>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefI16(%a : tensor<*xi16>, %b : tensor<*xi16>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefI32(%a : tensor<*xi32>, %b : tensor<*xi32>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefI64(%a : tensor<*xi64>, %b : tensor<*xi64>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefBF16(%a : tensor<*xbf16>, %b : tensor<*xbf16>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefF16(%a : tensor<*xf16>, %b : tensor<*xf16>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefF32(%a : tensor<*xf32>, %b : tensor<*xf32>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefF64(%a : tensor<*xf64>, %b : tensor<*xf64>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefC32(%a : tensor<*xcomplex<f32>>, %b : tensor<*xcomplex<f32>>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefC64(%a : tensor<*xcomplex<f64>>, %b : tensor<*xcomplex<f64>>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @verifyMemRefInd(%a : tensor<*xindex>, %b : tensor<*xindex>) -> i64 attributes { llvm.emit_c_interface }

  func.func @entry() {
    %i8 = arith.constant dense<90> : tensor<3x3xi8>
    %i16 = arith.constant dense<1> : tensor<3x3xi16>
    %i32 = arith.constant dense<2> : tensor<3x3xi32>
    %i64 = arith.constant dense<3> : tensor<3x3xi64>
    %f16 = arith.constant dense<1.5> : tensor<3x3xf16>
    %bf16 = arith.constant dense<2.5> : tensor<3x3xbf16>
    %f32 = arith.constant dense<3.5> : tensor<3x3xf32>
    %f64 = arith.constant dense<4.5> : tensor<3x3xf64>
    %c32 = arith.constant dense<(1.000000e+01,5.000000e+00)> : tensor<3x3xcomplex<f32>>
    %c64 = arith.constant dense<(2.000000e+01,5.000000e+00)> : tensor<3x3xcomplex<f64>>
    %ind = arith.constant dense<4> : tensor<3x3xindex>

    %1 = tensor.cast %i8 : tensor<3x3xi8> to tensor<*xi8>
    %2 = tensor.cast %i16 : tensor<3x3xi16> to tensor<*xi16>
    %3 = tensor.cast %i32 : tensor<3x3xi32> to tensor<*xi32>
    %4 = tensor.cast %i64 : tensor<3x3xi64> to tensor<*xi64>
    %5 = tensor.cast %f16 : tensor<3x3xf16> to tensor<*xf16>
    %6 = tensor.cast %bf16 : tensor<3x3xbf16> to tensor<*xbf16>
    %7 = tensor.cast %f32 : tensor<3x3xf32> to tensor<*xf32>
    %8 = tensor.cast %f64 : tensor<3x3xf64> to tensor<*xf64>
    %9 = tensor.cast %c32 : tensor<3x3xcomplex<f32>> to tensor<*xcomplex<f32>>
    %10 = tensor.cast %c64 : tensor<3x3xcomplex<f64>> to tensor<*xcomplex<f64>>
    %11 = tensor.cast %ind : tensor<3x3xindex> to tensor<*xindex>

    //
    // Ensure that verifyMemRef could detect equal memrefs.
    //
    // CHECK: 0
    %res0 = call @verifyMemRefI8(%1, %1) : (tensor<*xi8>, tensor<*xi8>) -> (i64)
    vector.print %res0 : i64

    // CHECK-NEXT: 0
    %res1 = call @verifyMemRefI16(%2, %2) : (tensor<*xi16>, tensor<*xi16>) -> (i64)
    vector.print %res1 : i64

    // CHECK-NEXT: 0
    %res2 = call @verifyMemRefI32(%3, %3) : (tensor<*xi32>, tensor<*xi32>) -> (i64)
    vector.print %res2 : i64

    // CHECK-NEXT: 0
    %res3 = call @verifyMemRefI64(%4, %4) : (tensor<*xi64>, tensor<*xi64>) -> (i64)
    vector.print %res3 : i64

    // CHECK-NEXT: 0
    %res4 = call @verifyMemRefF16(%5, %5) : (tensor<*xf16>, tensor<*xf16>) -> (i64)
    vector.print %res4 : i64

    // CHECK-NEXT: 0
    %res5 = call @verifyMemRefBF16(%6, %6) : (tensor<*xbf16>, tensor<*xbf16>) -> (i64)
    vector.print %res5 : i64

    // CHECK-NEXT: 0
    %res6 = call @verifyMemRefF32(%7, %7) : (tensor<*xf32>, tensor<*xf32>) -> (i64)
    vector.print %res6 : i64

    // CHECK-NEXT: 0
    %res7 = call @verifyMemRefF64(%8, %8) : (tensor<*xf64>, tensor<*xf64>) -> (i64)
    vector.print %res7 : i64

    // CHECK-NEXT: 0
    %res8 = call @verifyMemRefC32(%9, %9) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (i64)
    vector.print %res8 : i64

    // CHECK-NEXT: 0
    %res9 = call @verifyMemRefC64(%10, %10) : (tensor<*xcomplex<f64>>, tensor<*xcomplex<f64>>) -> (i64)
    vector.print %res9 : i64

    // CHECK-NEXT: 0
    %res10 = call @verifyMemRefInd(%11, %11) : (tensor<*xindex>, tensor<*xindex>) -> (i64)
    vector.print %res10 : i64

    //
    // Ensure that verifyMemRef could detect the correct number of errors
    // for unequal memrefs.
    //
    %m1 = arith.constant dense<100> : tensor<3x3xi8>
    %f1 = tensor.cast %m1 : tensor<3x3xi8> to tensor<*xi8>
    %fail_res1 = call @verifyMemRefI8(%1, %f1) : (tensor<*xi8>, tensor<*xi8>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res1 : i64

    %m2 = arith.constant dense<100> : tensor<3x3xi16>
    %f2 = tensor.cast %m2 : tensor<3x3xi16> to tensor<*xi16>
    %fail_res2 = call @verifyMemRefI16(%2, %f2) : (tensor<*xi16>, tensor<*xi16>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res2 : i64

    %m3 = arith.constant dense<100> : tensor<3x3xi32>
    %f3 = tensor.cast %m3 : tensor<3x3xi32> to tensor<*xi32>
    %fail_res3 = call @verifyMemRefI32(%3, %f3) : (tensor<*xi32>, tensor<*xi32>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res3 : i64

    %m4 = arith.constant dense<100> : tensor<3x3xi64>
    %f4 = tensor.cast %m4 : tensor<3x3xi64> to tensor<*xi64>
    %fail_res4 = call @verifyMemRefI64(%4, %f4) : (tensor<*xi64>, tensor<*xi64>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res4 : i64

    %m5 = arith.constant dense<100.0> : tensor<3x3xf16>
    %f5 = tensor.cast %m5 : tensor<3x3xf16> to tensor<*xf16>
    %fail_res5 = call @verifyMemRefF16(%5, %f5) : (tensor<*xf16>, tensor<*xf16>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res5 : i64

    %m6 = arith.constant dense<100.0> : tensor<3x3xbf16>
    %f6 = tensor.cast %m6 : tensor<3x3xbf16> to tensor<*xbf16>
    %fail_res6 = call @verifyMemRefBF16(%6, %f6) : (tensor<*xbf16>, tensor<*xbf16>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res6 : i64

    %m7 = arith.constant dense<100.0> : tensor<3x3xf32>
    %f7 = tensor.cast %m7 : tensor<3x3xf32> to tensor<*xf32>
    %fail_res7 = call @verifyMemRefF32(%7, %f7) : (tensor<*xf32>, tensor<*xf32>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res7 : i64

    %m8 = arith.constant dense<100.0> : tensor<3x3xf64>
    %f8 = tensor.cast %m8 : tensor<3x3xf64> to tensor<*xf64>
    %fail_res8 = call @verifyMemRefF64(%8, %f8) : (tensor<*xf64>, tensor<*xf64>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res8 : i64

    %m9 = arith.constant dense<(5.000000e+01,1.000000e+00)> : tensor<3x3xcomplex<f32>>
    %f9 = tensor.cast %m9 : tensor<3x3xcomplex<f32>> to tensor<*xcomplex<f32>>
    %fail_res9 = call @verifyMemRefC32(%9, %f9) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res9 : i64

    %m10 = arith.constant dense<(5.000000e+01,1.000000e+00)> : tensor<3x3xcomplex<f64>>
    %f10 = tensor.cast %m10 : tensor<3x3xcomplex<f64>> to tensor<*xcomplex<f64>>
    %fail_res10 = call @verifyMemRefC64(%10, %f10) : (tensor<*xcomplex<f64>>, tensor<*xcomplex<f64>>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res10 : i64

    %m11 = arith.constant dense<100> : tensor<3x3xindex>
    %f11 = tensor.cast %m11 : tensor<3x3xindex> to tensor<*xindex>
    %fail_res11 = call @verifyMemRefInd(%11, %f11) : (tensor<*xindex>, tensor<*xindex>) -> (i64)
    // CHECK-NEXT: 9
    vector.print %fail_res11 : i64

    return
  }
}
