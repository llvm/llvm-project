// RUN: mlir-opt %s \
// RUN:   -func-bufferize -tensor-bufferize -arith-bufferize --canonicalize \
// RUN:   -convert-scf-to-cf --convert-complex-to-standard \
// RUN:   -convert-memref-to-llvm -convert-math-to-llvm -convert-math-to-libm \
// RUN:   -convert-vector-to-llvm -convert-complex-to-llvm \
// RUN:   -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext |\
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
  func.return
}
