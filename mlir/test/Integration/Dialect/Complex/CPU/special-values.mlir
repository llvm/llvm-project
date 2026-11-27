// RUN: mlir-opt %s \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" --canonicalize \
// RUN:   -convert-scf-to-cf --convert-complex-to-standard \
// RUN:   -finalize-memref-to-llvm -convert-math-to-llvm -convert-math-to-libm \
// RUN:   -convert-vector-to-llvm -convert-complex-to-llvm \
// RUN:   -convert-func-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:   -reconcile-unrealized-casts |\
// RUN: mlir-runner \
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

func.func @exp(%arg: complex<f32>) -> complex<f32> {
  %exp = complex.exp %arg : complex<f32>
  func.return %exp : complex<f32>
}

func.func @sqrt(%arg: complex<f32>) -> complex<f32> {
  %sqrt = complex.sqrt %arg : complex<f32>
  func.return %sqrt : complex<f32>
}

func.func @tanh(%arg: complex<f32>) -> complex<f32> {
  %tanh = complex.tanh %arg : complex<f32>
  func.return %tanh : complex<f32>
}

func.func @entry() {
  // complex.exp: C99 G.6.3.1
  %exp_test = arith.constant dense<[
    // cexp(0 + i*0) = 1 + i*0
    (0x00000000, 0x00000000),
    // CHECK:      {{^}}1{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // cexp(0 - i*0) = 1 - i*0
    (0x00000000, 0x80000000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // cexp(-0 + i*0) = 1 + i*0
    (0x80000000, 0x00000000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // cexp(-0 - i*0) = 1 - i*0
    (0x80000000, 0x80000000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // cexp(2 + i*inf) = nan + i*nan
    (0x40000000, 0x7F800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(2 - i*inf) = nan + i*nan
    (0x40000000, 0xFF800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(2 + i*nan) = nan + i*nan
    (0x40000000, 0x7FC00000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(-2 + i*inf) = nan + i*nan
    (0xC0000000, 0x7F800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(-2 - i*inf) = nan + i*nan
    (0xC0000000, 0xFF800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(0 + i*inf) = nan + i*nan
    (0x00000000, 0x7F800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(0 - i*inf) = nan + i*nan
    (0x00000000, 0xFF800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(0 + i*nan) = nan + i*nan
    (0x00000000, 0x7FC00000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(inf + i*0) = inf + i*0
    (0x7F800000, 0x00000000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // cexp(inf - i*0) = inf - i*0
    (0x7F800000, 0x80000000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // cexp(-inf + i*0) = +0 + i*0
    (0xFF800000, 0x00000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // cexp(-inf - i*0) = +0 - i*0
    (0xFF800000, 0x80000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // cexp(-inf + i*2) = -0 + i*0
    (0xFF800000, 0x40000000),
    // CHECK-NEXT: {{^}}-0{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // cexp(-inf - i*2) = -0 - i*0
    (0xFF800000, 0xC0000000),
    // CHECK-NEXT: {{^}}-0{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // cexp(-inf + i*0.5) = +0 + i*0
    (0xFF800000, 0x3F000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // cexp(-inf - i*0.5) = +0 - i*0
    (0xFF800000, 0xBF000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // cexp(inf + i*2) = -inf + i*inf
    (0x7F800000, 0x40000000),
    // CHECK-NEXT: {{^}}-inf{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // cexp(inf - i*2) = -inf - i*inf
    (0x7F800000, 0xC0000000),
    // CHECK-NEXT: {{^}}-inf{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // cexp(inf + i*0.5) = inf + i*inf
    (0x7F800000, 0x3F000000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // cexp(inf - i*0.5) = inf - i*inf
    (0x7F800000, 0xBF000000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // cexp(-inf + i*inf) = +-0 +- i*0
    (0xFF800000, 0x7F800000),
    // CHECK-NEXT: {{^-?0$}}
    // CHECK-NEXT: {{^-?0$}}
    // cexp(-inf - i*inf) = +-0 +- i*0
    (0xFF800000, 0xFF800000),
    // CHECK-NEXT: {{^-?0$}}
    // CHECK-NEXT: {{^-?0$}}
    // cexp(inf + i*inf) = +-inf + i*nan
    (0x7F800000, 0x7F800000),
    // CHECK-NEXT: {{^-?inf$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(inf - i*inf) = +-inf + i*nan
    (0x7F800000, 0xFF800000),
    // CHECK-NEXT: {{^-?inf$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(-inf + i*nan) = +-0 +- i*0
    (0xFF800000, 0x7FC00000),
    // CHECK-NEXT: {{^-?0$}}
    // CHECK-NEXT: {{^-?0$}}
    // cexp(inf + i*nan) = +-inf + i*nan
    (0x7F800000, 0x7FC00000),
    // CHECK-NEXT: {{^-?inf$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(nan + i*0) = nan + i*0
    (0x7FC00000, 0x00000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // cexp(nan - i*0) = nan - i*0
    (0x7FC00000, 0x80000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // cexp(nan + i*2) = nan + i*nan
    (0x7FC00000, 0x40000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(nan - i*2) = nan + i*nan
    (0x7FC00000, 0xC0000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(nan + i*inf) = nan + i*nan
    (0x7FC00000, 0x7F800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(nan - i*inf) = nan + i*nan
    (0x7FC00000, 0xFF800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // cexp(nan + i*nan) = nan + i*nan
    (0x7FC00000, 0x7FC00000)
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
  ]> : tensor<37xcomplex<f32>>
  %exp_test_cast = tensor.cast %exp_test
    :  tensor<37xcomplex<f32>> to tensor<?xcomplex<f32>>

  %exp_func = func.constant @exp : (complex<f32>) -> complex<f32>
  call @test_unary(%exp_test_cast, %exp_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>) -> complex<f32>) -> ()

  // complex.sqrt: C99 G.6.4.2
  // FIXME: cases commented out currently fail
  %sqrt_test = arith.constant dense<[
    // csqrt(0 + i*0) = +0 + i*0
    (0x00000000, 0x00000000),
    // CHECK:      {{^}}0{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // csqrt(0 - i*0) = +0 - i*0
    // (0x00000000, 0x80000000),
    // COM: CHECK-NEXT: {{^}}0{{$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // csqrt(-0 + i*0) = +0 + i*0
    (0x80000000, 0x00000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // csqrt(-0 - i*0) = +0 - i*0
    // (0x80000000, 0x80000000),
    // COM: CHECK-NEXT: {{^}}0{{$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // csqrt(2 + i*inf) = inf + i*inf
    (0x40000000, 0x7F800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // csqrt(2 - i*inf) = inf - i*inf
    (0x40000000, 0xFF800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // csqrt(-2 + i*inf) = inf + i*inf
    (0xC0000000, 0x7F800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // csqrt(-2 - i*inf) = inf - i*inf
    (0xC0000000, 0xFF800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // csqrt(0 + i*inf) = inf + i*inf
    (0x00000000, 0x7F800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // csqrt(0 - i*inf) = inf - i*inf
    (0x00000000, 0xFF800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // csqrt(inf + i*inf) = inf + i*inf
    (0x7F800000, 0x7F800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // csqrt(inf - i*inf) = inf - i*inf
    (0x7F800000, 0xFF800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // csqrt(-inf + i*inf) = inf + i*inf
    (0xFF800000, 0x7F800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // csqrt(-inf - i*inf) = inf - i*inf
    (0xFF800000, 0xFF800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // csqrt(nan + i*inf) = inf + i*inf
    (0x7FC00000, 0x7F800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // csqrt(nan - i*inf) = inf - i*inf
    (0x7FC00000, 0xFF800000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // csqrt(2 + i*nan) = nan + i*nan
    (0x40000000, 0x7FC00000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // csqrt(0 + i*nan) = nan + i*nan
    (0x00000000, 0x7FC00000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // csqrt(-inf + i*2) = +0 + i*inf
    (0xFF800000, 0x40000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // csqrt(-inf - i*2) = +0 - i*inf
    (0xFF800000, 0xC0000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // csqrt(inf + i*2) = inf + i*0
    (0x7F800000, 0x40000000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // csqrt(inf - i*2) = inf - i*0
    // (0x7F800000, 0xC0000000),
    // COM: CHECK-NEXT: {{^}}inf{{$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // csqrt(-inf + i*0) = +0 + i*inf
    (0xFF800000, 0x00000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}inf{{$}}
    // csqrt(-inf - i*0) = +0 - i*inf
    (0xFF800000, 0x80000000),
    // CHECK-NEXT: {{^}}0{{$}}
    // CHECK-NEXT: {{^}}-inf{{$}}
    // csqrt(inf + i*0) = inf + i*0
    (0x7F800000, 0x00000000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // csqrt(inf - i*0) = inf - i*0
    // (0x7F800000, 0x80000000),
    // COM: CHECK-NEXT: {{^}}inf{{$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // csqrt(-inf + i*nan) = nan +- i*inf
    (0xFF800000, 0x7FC00000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?inf$}}
    // csqrt(inf + i*nan) = inf + i*nan
    (0x7F800000, 0x7FC00000),
    // CHECK-NEXT: {{^}}inf{{$}}
    // CHECK-NEXT: {{^-?nan$}}
    // csqrt(nan + i*2) = nan + i*nan
    (0x7FC00000, 0x40000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // csqrt(nan - i*2) = nan + i*nan
    (0x7FC00000, 0xC0000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // csqrt(nan + i*0) = nan + i*nan
    (0x7FC00000, 0x00000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // csqrt(nan - i*0) = nan + i*nan
    (0x7FC00000, 0x80000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // csqrt(nan + i*nan) = nan + i*nan
    (0x7FC00000, 0x7FC00000)
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
  ]> : tensor<29xcomplex<f32>>
  %sqrt_test_cast = tensor.cast %sqrt_test
    :  tensor<29xcomplex<f32>> to tensor<?xcomplex<f32>>

  %sqrt_func = func.constant @sqrt : (complex<f32>) -> complex<f32>
  call @test_unary(%sqrt_test_cast, %sqrt_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>) -> complex<f32>) -> ()

  // complex.tanh: C17 G.6.2.6
  // FIXME: cases commented out currently fail
  %tanh_test = arith.constant dense<[
    // ctanh(0 + i*0) = +0 + i*0
    (0x00000000, 0x00000000),
    // CHECK:      {{^}}0{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // ctanh(0 - i*0) = +0 - i*0
    // (0x00000000, 0x80000000),
    // COM: CHECK-NEXT: {{^}}0{{$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(-0 + i*0) = -0 + i*0
    // (0x80000000, 0x00000000),
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // COM: CHECK-NEXT: {{^}}0{{$}}
    // ctanh(-0 - i*0) = -0 - i*0
    // (0x80000000, 0x80000000),
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(2 + i*inf) = nan + i*nan
    (0x40000000, 0x7F800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(2 - i*inf) = nan + i*nan
    (0x40000000, 0xFF800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(-2 + i*inf) = nan + i*nan
    (0xC0000000, 0x7F800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(-2 - i*inf) = nan + i*nan
    (0xC0000000, 0xFF800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(2 + i*nan) = nan + i*nan
    (0x40000000, 0x7FC00000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(-2 + i*nan) = nan + i*nan
    (0xC0000000, 0x7FC00000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(0 + i*inf) = +0 + i*nan
    // (0x00000000, 0x7F800000),
    // COM: CHECK-NEXT: {{^}}0{{$}}
    // COM: CHECK-NEXT: {{^-?nan$}}
    // ctanh(0 - i*inf) = +0 + i*nan
    // (0x00000000, 0xFF800000),
    // COM: CHECK-NEXT: {{^}}0{{$}}
    // COM: CHECK-NEXT: {{^-?nan$}}
    // ctanh(-0 + i*inf) = -0 + i*nan
    // (0x80000000, 0x7F800000),
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // COM: CHECK-NEXT: {{^-?nan$}}
    // ctanh(-0 - i*inf) = -0 + i*nan
    // (0x80000000, 0xFF800000),
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // COM: CHECK-NEXT: {{^-?nan$}}
    // ctanh(0 + i*nan) = +0 + i*nan
    // (0x00000000, 0x7FC00000),
    // COM: CHECK-NEXT: {{^}}0{{$}}
    // COM: CHECK-NEXT: {{^-?nan$}}
    // ctanh(-0 + i*nan) = -0 + i*nan
    // (0x80000000, 0x7FC00000),
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // COM: CHECK-NEXT: {{^-?nan$}}
    // ctanh(inf + i*0) = 1 + i*0
    (0x7F800000, 0x00000000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // ctanh(inf - i*0) = 1 - i*0
    // (0x7F800000, 0x80000000),
    // COM: CHECK-NEXT: {{^}}1{{$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(-inf + i*0) = -1 + i*0
    (0xFF800000, 0x00000000),
    // CHECK-NEXT: {{^}}-1{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // ctanh(-inf - i*0) = -1 - i*0
    // (0xFF800000, 0x80000000),
    // COM: CHECK-NEXT: {{^}}-1{{$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(inf + i*0.5) = 1 + i*0
    (0x7F800000, 0x3F000000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // ctanh(inf - i*0.5) = 1 - i*0
    (0x7F800000, 0xBF000000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(-inf + i*0.5) = -1 + i*0
    (0xFF800000, 0x3F000000),
    // CHECK-NEXT: {{^}}-1{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // ctanh(-inf - i*0.5) = -1 - i*0
    (0xFF800000, 0xBF000000),
    // CHECK-NEXT: {{^}}-1{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(inf + i*2) = 1 - i*0
    (0x7F800000, 0x40000000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(inf - i*2) = 1 + i*0
    (0x7F800000, 0xC0000000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // ctanh(-inf + i*2) = -1 - i*0
    (0xFF800000, 0x40000000),
    // CHECK-NEXT: {{^}}-1{{$}}
    // CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(-inf - i*2) = -1 + i*0
    (0xFF800000, 0xC0000000),
    // CHECK-NEXT: {{^}}-1{{$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // ctanh(inf + i*inf) = 1 +- i*0
    (0x7F800000, 0x7F800000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^-?0$}}
    // ctanh(inf - i*inf) = 1 +- i*0
    (0x7F800000, 0xFF800000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^-?0$}}
    // ctanh(-inf + i*inf) = -1 +- i*0
    (0xFF800000, 0x7F800000),
    // CHECK-NEXT: {{^}}-1{{$}}
    // CHECK-NEXT: {{^-?0$}}
    // ctanh(-inf - i*inf) = -1 +- i*0
    (0xFF800000, 0xFF800000),
    // CHECK-NEXT: {{^}}-1{{$}}
    // CHECK-NEXT: {{^-?0$}}
    // ctanh(inf + i*nan) = 1 +- i*0
    (0x7F800000, 0x7FC00000),
    // CHECK-NEXT: {{^}}1{{$}}
    // CHECK-NEXT: {{^-?0$}}
    // ctanh(-inf + i*nan) = -1 +- i*0
    (0xFF800000, 0x7FC00000),
    // CHECK-NEXT: {{^}}-1{{$}}
    // CHECK-NEXT: {{^-?0$}}
    // ctanh(nan + i*0) = nan + i*0
    (0x7FC00000, 0x00000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^}}0{{$}}
    // ctanh(nan - i*0) = nan - i*0
    // (0x7FC00000, 0x80000000),
    // COM: CHECK-NEXT: {{^-?nan$}}
    // COM: CHECK-NEXT: {{^}}-0{{$}}
    // ctanh(nan + i*2) = nan + i*nan
    (0x7FC00000, 0x40000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(nan - i*2) = nan + i*nan
    (0x7FC00000, 0xC0000000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(nan + i*inf) = nan + i*nan
    (0x7FC00000, 0x7F800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(nan - i*inf) = nan + i*nan
    (0x7FC00000, 0xFF800000),
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
    // ctanh(nan + i*nan) = nan + i*nan
    (0x7FC00000, 0x7FC00000)
    // CHECK-NEXT: {{^-?nan$}}
    // CHECK-NEXT: {{^-?nan$}}
  ]> : tensor<29xcomplex<f32>>
  %tanh_test_cast = tensor.cast %tanh_test
    :  tensor<29xcomplex<f32>> to tensor<?xcomplex<f32>>

  %tanh_func = func.constant @tanh : (complex<f32>) -> complex<f32>
  call @test_unary(%tanh_test_cast, %tanh_func)
    : (tensor<?xcomplex<f32>>, (complex<f32>) -> complex<f32>) -> ()

  func.return
}

