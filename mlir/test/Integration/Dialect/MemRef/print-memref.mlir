// RUN: mlir-opt %s \
// RUN: -one-shot-bufferize="bufferize-function-boundaries" --canonicalize \
// RUN:   -finalize-memref-to-llvm\
// RUN:   -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils |\
// RUN: FileCheck %s

module {

  func.func private @printMemrefI8(%ptr : tensor<*xi8>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefI16(%ptr : tensor<*xi16>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefI32(%ptr : tensor<*xi32>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefI64(%ptr : tensor<*xi64>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefBF16(%ptr : tensor<*xbf16>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefF16(%ptr : tensor<*xf16>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefF32(%ptr : tensor<*xf32>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefF64(%ptr : tensor<*xf64>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefC32(%ptr : tensor<*xcomplex<f32>>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefC64(%ptr : tensor<*xcomplex<f64>>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefInd(%ptr : tensor<*xindex>) attributes { llvm.emit_c_interface }

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

    // CHECK:      data = 
    // CHECK-NEXT: {{\[}}[Z,   Z,   Z],
    // CHECK-NEXT:       [Z,   Z,   Z],
    // CHECK-NEXT:       [Z,   Z,   Z]]
    //
    call @printMemrefI8(%1) : (tensor<*xi8>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[1,   1,   1],
    // CHECK-NEXT:       [1,   1,   1],
    // CHECK-NEXT:       [1,   1,   1]]
    //
    call @printMemrefI16(%2) : (tensor<*xi16>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[2,   2,   2],
    // CHECK-NEXT:       [2,   2,   2],
    // CHECK-NEXT:       [2,   2,   2]]
    //
    call @printMemrefI32(%3) : (tensor<*xi32>) -> ()

    // CHECK-NEXT: data =
    // CHECK-NEXT: {{\[}}[3,   3,   3],
    // CHECK-NEXT:       [3,   3,   3],
    // CHECK-NEXT:       [3,   3,   3]]
    //
    call @printMemrefI64(%4) : (tensor<*xi64>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[1.5,   1.5,   1.5],
    // CHECK-NEXT:       [1.5,   1.5,   1.5],
    // CHECK-NEXT:       [1.5,   1.5,   1.5]]
    //
    call @printMemrefF16(%5) : (tensor<*xf16>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[2.5,   2.5,   2.5],
    // CHECK-NEXT:       [2.5,   2.5,   2.5],
    // CHECK-NEXT:       [2.5,   2.5,   2.5]]
    //
    call @printMemrefBF16(%6) : (tensor<*xbf16>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[3.5,   3.5,   3.5],
    // CHECK-NEXT:       [3.5,   3.5,   3.5],
    // CHECK-NEXT:       [3.5,   3.5,   3.5]]
    //
    call @printMemrefF32(%7) : (tensor<*xf32>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[4.5,   4.5,   4.5],
    // CHECK-NEXT:       [4.5,   4.5,   4.5],
    // CHECK-NEXT:       [4.5,   4.5,   4.5]]
    //
    call @printMemrefF64(%8) : (tensor<*xf64>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[(10,5), (10,5), (10,5)],
    // CHECK-NEXT:       [(10,5), (10,5), (10,5)],
    // CHECK-NEXT:       [(10,5), (10,5), (10,5)]]
    //
    call @printMemrefC32(%9) : (tensor<*xcomplex<f32>>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[(20,5), (20,5), (20,5)],
    // CHECK-NEXT:       [(20,5), (20,5), (20,5)],
    // CHECK-NEXT:       [(20,5), (20,5), (20,5)]]
    //
    call @printMemrefC64(%10) : (tensor<*xcomplex<f64>>) -> ()

    // CHECK-NEXT: data = 
    // CHECK-NEXT: {{\[}}[4,   4,   4],
    // CHECK-NEXT:       [4,   4,   4],
    // CHECK-NEXT:       [4,   4,   4]]
    //
    call @printMemrefInd(%11) : (tensor<*xindex>) -> ()

    return
  }
}
