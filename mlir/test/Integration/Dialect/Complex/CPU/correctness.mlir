// RUN: mlir-opt %s \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" --canonicalize \
// RUN:   -convert-scf-to-cf --convert-complex-to-standard \
// RUN:   -finalize-memref-to-llvm -convert-math-to-llvm -convert-math-to-libm \
// RUN:   -convert-vector-to-llvm -convert-complex-to-llvm \
// RUN:   -convert-func-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:   -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_c_runner_utils |\
// RUN: FileCheck %s

func.func @test_unary(%input: tensor<?xcomplex<f32>>,
                      %func: (complex<f32>) -> complex<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %size = tensor.dim %input, %c0: tensor<?xcomplex<f32>>

  scf.for %i = %c0 to %size step %c1 {
    %elem = tensor.extract %input[%i]: tensor<?xcomplex<f32>>

    %val = func.call_indirect %func(%elem) : (complex<f32>) -> complex<f32>
    %real = complex.re %val : complex<f32>
    %imag = complex.im %val: complex<f32>
    vector.print %real : f32
    vector.print %imag : f32
    scf.yield
  }
  func.return
}

func.func @sqrt(%arg: complex<f32>) -> complex<f32> {
  %sqrt = complex.sqrt %arg : complex<f32>
  func.return %sqrt : complex<f32>
}

func.func @tanh(%arg: complex<f32>) -> complex<f32> {
  %tanh = complex.tanh %arg : complex<f32>
  func.return %tanh : complex<f32>
}

func.func @rsqrt(%arg: complex<f32>) -> complex<f32> {
  %sqrt = complex.rsqrt %arg : complex<f32>
  func.return %sqrt : complex<f32>
}

func.func @conj(%arg: complex<f32>) -> complex<f32> {
  %conj = complex.conj %arg : complex<f32>
  func.return %conj : complex<f32>
}

// %input contains pairs of lhs, rhs, i.e. [lhs_0, rhs_0, lhs_1, rhs_1,...]
func.func @test_binary(%input: tensor<?xcomplex<f32>>,
                       %func: (complex<f32>, complex<f32>) -> complex<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %size = tensor.dim %input, %c0: tensor<?xcomplex<f32>>

  scf.for %i = %c0 to %size step %c2 {
    %lhs = tensor.extract %input[%i]: tensor<?xcomplex<f32>>
    %i_next = arith.addi %i, %c1 : index
    %rhs = tensor.extract %input[%i_next]: tensor<?xcomplex<f32>>

    %val = func.call_indirect %func(%lhs, %rhs)
      : (complex<f32>, complex<f32>) -> complex<f32>
    %real = complex.re %val : complex<f32>
    %imag = complex.im %val: complex<f32>
    vector.print %real : f32
    vector.print %imag : f32
    scf.yield
  }
  func.return
}

func.func @atan2(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %atan2 = complex.atan2 %lhs, %rhs : complex<f32>
  func.return %atan2 : complex<f32>
}

func.func @pow(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %pow = complex.pow %lhs, %rhs : complex<f32>
  func.return %pow : complex<f32>
}

func.func @test_element(%input: tensor<?xcomplex<f32>>,
                      %func: (complex<f32>) -> f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %size = tensor.dim %input, %c0: tensor<?xcomplex<f32>>

  scf.for %i = %c0 to %size step %c1 {
    %elem = tensor.extract %input[%i]: tensor<?xcomplex<f32>>

    %val = func.call_indirect %func(%elem) : (complex<f32>) -> f32
    vector.print %val : f32
    scf.yield
  }
  func.return
}

func.func @angle(%arg: complex<f32>) -> f32 {
  %angle = complex.angle %arg : complex<f32>
  func.return %angle : f32
}

func.func @test_element_f64(%input: tensor<?xcomplex<f64>>,
                      %func: (complex<f64>) -> f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %size = tensor.dim %input, %c0: tensor<?xcomplex<f64>>

  scf.for %i = %c0 to %size step %c1 {
    %elem = tensor.extract %input[%i]: tensor<?xcomplex<f64>>

    %val = func.call_indirect %func(%elem) : (complex<f64>) -> f64
    vector.print %val : f64
    scf.yield
  }
  func.return
}

func.func @abs(%arg: complex<f64>) -> f64 {
  %abs = complex.abs %arg : complex<f64>
  func.return %abs : f64
}

func.func @entry() {
  // complex.sqrt test
  %sqrt_test = arith.constant dense<[
    (-1.0, -1.0),
    // CHECK:       0.455
    // CHECK-NEXT: -1.098
    (-1.0, 1.0),
    // CHECK-NEXT:  0.455
    // CHECK-NEXT:  1.098
    (0.0, 0.0),
    // CHECK-NEXT:  0
    // CHECK-NEXT:  0
    (0.0, 1.0),
    // CHECK-NEXT:  0.707
    // CHECK-NEXT:  0.707
    (1.0, -1.0),
    // CHECK-NEXT:  1.098
    // CHECK-NEXT:  -0.455
    (1.0, 0.0),
    // CHECK-NEXT:  1
    // CHECK-NEXT:  0
    (1.0, 1.0)
    // CHECK-NEXT:  1.098
    // CHECK-NEXT:  0.455
  ]> : tensor<7xcomplex<f32>>
  %sqrt_test_cast = tensor.cast %sqrt_test
    :  tensor<7xcomplex<f32>> to tensor<?xcomplex<f32>>

  %sqrt_func = func.constant @sqrt : (complex<f32>) -> complex<f32>
  call @test_unary(%sqrt_test_cast, %sqrt_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>) -> complex<f32>) -> ()

  // complex.atan2 test
  %atan2_test = arith.constant dense<[
    (1.0, 2.0), (2.0, 1.0),
    // CHECK:       0.785
    // CHECK-NEXT:  0.346
    (1.0, 1.0), (1.0, 0.0),
    // CHECK-NEXT:  1.017
    // CHECK-NEXT:  0.402
    (1.0, 1.0), (1.0, 1.0)
    // CHECK-NEXT:  0.785
    // CHECK-NEXT:  0
  ]> : tensor<6xcomplex<f32>>
  %atan2_test_cast = tensor.cast %atan2_test
    :  tensor<6xcomplex<f32>> to tensor<?xcomplex<f32>>

  %atan2_func = func.constant @atan2 : (complex<f32>, complex<f32>)
    -> complex<f32>
  call @test_binary(%atan2_test_cast, %atan2_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>, complex<f32>)
    -> complex<f32>) -> ()

  // complex.pow test
  %pow_test = arith.constant dense<[
    (0.0, 0.0), (0.0, 0.0),
    // CHECK:       1
    // CHECK-NEXT:  0
    (0.0, 0.0), (1.0, 0.0),
    // CHECK-NEXT:  0
    // CHECK-NEXT:  0
    (0.0, 0.0), (-1.0, 0.0),
    // Ignoring the sign of nan as that can't be tested in platform agnostic manner. See: #58531
    // CHECK-NEXT:  nan
    // CHECK-NEXT:  nan
    (1.0, 1.0), (1.0, 1.0)
    // CHECK-NEXT:  0.273
    // CHECK-NEXT:  0.583
  ]> : tensor<8xcomplex<f32>>
  %pow_test_cast = tensor.cast %pow_test
    :  tensor<8xcomplex<f32>> to tensor<?xcomplex<f32>>

  %pow_func = func.constant @pow : (complex<f32>, complex<f32>)
    -> complex<f32>
  call @test_binary(%pow_test_cast, %pow_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>, complex<f32>)
    -> complex<f32>) -> ()

  // complex.tanh test
  %tanh_test = arith.constant dense<[
    (-1.0, -1.0),
    // CHECK:      -1.08392
    // CHECK-NEXT: -0.271753
    (-1.0, 1.0),
    // CHECK-NEXT:  -1.08392
    // CHECK-NEXT:  0.271753
    (0.0, 0.0),
    // CHECK-NEXT:  0
    // CHECK-NEXT:  0
    (0.0, 1.0),
    // CHECK-NEXT:  0
    // CHECK-NEXT:  1.5574
    (1.0, -1.0),
    // CHECK-NEXT:  1.08392
    // CHECK-NEXT:  -0.271753
    (1.0, 0.0),
    // CHECK-NEXT:  0.761594
    // CHECK-NEXT:  0
    (1.0, 1.0)
    // CHECK-NEXT:  1.08392
    // CHECK-NEXT:  0.271753
  ]> : tensor<7xcomplex<f32>>
  %tanh_test_cast = tensor.cast %tanh_test
    :  tensor<7xcomplex<f32>> to tensor<?xcomplex<f32>>

  %tanh_func = func.constant @tanh : (complex<f32>) -> complex<f32>
  call @test_unary(%tanh_test_cast, %tanh_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>) -> complex<f32>) -> ()

  // complex.rsqrt test
  %rsqrt_test = arith.constant dense<[
    (-1.0, -1.0),
    // CHECK:       0.321
    // CHECK-NEXT:  0.776
    (-1.0, 1.0),
    // CHECK-NEXT:  0.321
    // CHECK-NEXT:  -0.776
    (0.0, 0.0),
    // CHECK-NEXT:  inf
    // CHECK-NEXT:  nan
    (0.0, 1.0),
    // CHECK-NEXT:  0.707
    // CHECK-NEXT:  -0.707
    (1.0, -1.0),
    // CHECK-NEXT:  0.776
    // CHECK-NEXT:  0.321
    (1.0, 0.0),
    // CHECK-NEXT:  1
    // CHECK-NEXT:  0
    (1.0, 1.0)
    // CHECK-NEXT:  0.776
    // CHECK-NEXT:  -0.321
  ]> : tensor<7xcomplex<f32>>
  %rsqrt_test_cast = tensor.cast %rsqrt_test
    :  tensor<7xcomplex<f32>> to tensor<?xcomplex<f32>>

  %rsqrt_func = func.constant @rsqrt : (complex<f32>) -> complex<f32>
  call @test_unary(%rsqrt_test_cast, %rsqrt_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>) -> complex<f32>) -> ()

  // complex.conj test
  %conj_test = arith.constant dense<[
    (-1.0, -1.0),
    // CHECK:      -1
    // CHECK-NEXT: 1
    (-1.0, 1.0),
    // CHECK-NEXT:  -1
    // CHECK-NEXT:  -1
    (0.0, 0.0),
    // CHECK-NEXT:  0
    // CHECK-NEXT:  0
    (0.0, 1.0),
    // CHECK-NEXT:  0
    // CHECK-NEXT:  -1
    (1.0, -1.0),
    // CHECK-NEXT:  1
    // CHECK-NEXT:  1
    (1.0, 0.0),
    // CHECK-NEXT:  1
    // CHECK-NEXT:  0
    (1.0, 1.0)
    // CHECK-NEXT:  1
    // CHECK-NEXT:  -1
  ]> : tensor<7xcomplex<f32>>
  %conj_test_cast = tensor.cast %conj_test
    :  tensor<7xcomplex<f32>> to tensor<?xcomplex<f32>>

  %conj_func = func.constant @conj : (complex<f32>) -> complex<f32>
  call @test_unary(%conj_test_cast, %conj_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>) -> complex<f32>) -> ()

  // complex.angle test
  %angle_test = arith.constant dense<[
    (-1.0, -1.0),
    // CHECK:      -2.356
    (-1.0, 1.0),
    // CHECK-NEXT:  2.356
    (0.0, 0.0),
    // CHECK-NEXT:  0
    (0.0, 1.0),
    // CHECK-NEXT:  1.570
    (1.0, -1.0),
    // CHECK-NEXT:  -0.785
    (1.0, 0.0),
    // CHECK-NEXT:  0
    (1.0, 1.0)
    // CHECK-NEXT:  0.785
  ]> : tensor<7xcomplex<f32>>
  %angle_test_cast = tensor.cast %angle_test
    :  tensor<7xcomplex<f32>> to tensor<?xcomplex<f32>>

  %angle_func = func.constant @angle : (complex<f32>) -> f32
  call @test_element(%angle_test_cast, %angle_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>) -> f32) -> ()

  // complex.abs test
  %abs_test = arith.constant dense<[
    (1.0, 1.0),
    // CHECK:  1.414
    (1.0e300, 1.0e300),
    // CHECK-NEXT:  1.41421e+300
    (1.0e-300, 1.0e-300),
    // CHECK-NEXT:  1.41421e-300
    (5.0, 0.0),
    // CHECK-NEXT:  5
    (0.0, 6.0),
    // CHECK-NEXT:  6
    (7.0, 8.0),
    // CHECK-NEXT:  10.6301
    (-1.0, -1.0),
    // CHECK-NEXT: 1.414
    (-1.0e300, -1.0e300),
    // CHECK-NEXT:  1.41421e+300
    (-1.0, 0.0),
    // CHECK-NOT: -1
    // CHECK-NEXT:  1
    (0.0, -1.0)
    // CHECK-NOT:  -1
    // CHECK-NEXT:  1
  ]> : tensor<10xcomplex<f64>>
  %abs_test_cast = tensor.cast %abs_test
    :  tensor<10xcomplex<f64>> to tensor<?xcomplex<f64>>

  %abs_func = func.constant @abs : (complex<f64>) -> f64

  call @test_element_f64(%abs_test_cast, %abs_func)
    : (tensor<?xcomplex<f64>>, (complex<f64>) -> f64) -> ()

  func.return
}
