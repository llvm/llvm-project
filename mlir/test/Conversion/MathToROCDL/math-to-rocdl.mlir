// RUN: mlir-opt %s -convert-math-to-rocdl -allow-unregistered-dialect -split-input-file | FileCheck %s

module @test_module {
  // CHECK: llvm.func @__ocml_fmod_f16(f16, f16) -> f16
  // CHECK: llvm.func @__ocml_fmod_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_fmod_f64(f64, f64) -> f64
  // CHECK-LABEL: func @arith_remf
  func.func @arith_remf(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = arith.remf %arg_f16, %arg_f16 : f16
    // CHECK: llvm.call @__ocml_fmod_f16(%{{.*}}, %{{.*}}) : (f16, f16) -> f16
    %result32 = arith.remf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_fmod_f32(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = arith.remf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_fmod_f64(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_acos_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_acos_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_acos_f64(f64) -> f64
  // CHECK-LABEL: func @math_acos
  func.func @math_acos(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.acos %arg_f16 : f16
    // CHECK: llvm.call @__ocml_acos_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.acos %arg_f32 : f32
    // CHECK: llvm.call @__ocml_acos_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.acos %arg_f64 : f64
    // CHECK: llvm.call @__ocml_acos_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_acosh_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_acosh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_acosh_f64(f64) -> f64
  // CHECK-LABEL: func @math_acosh
  func.func @math_acosh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.acosh %arg_f16 : f16
    // CHECK: llvm.call @__ocml_acosh_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.acosh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_acosh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.acosh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_acosh_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_asin_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_asin_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_asin_f64(f64) -> f64
  // CHECK-LABEL: func @math_asin
  func.func @math_asin(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.asin %arg_f16 : f16
    // CHECK: llvm.call @__ocml_asin_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.asin %arg_f32 : f32
    // CHECK: llvm.call @__ocml_asin_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.asin %arg_f64 : f64
    // CHECK: llvm.call @__ocml_asin_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_asinh_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_asinh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_asinh_f64(f64) -> f64
  // CHECK-LABEL: func @math_asinh
  func.func @math_asinh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.asinh %arg_f16 : f16
    // CHECK: llvm.call @__ocml_asinh_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.asinh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_asinh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.asinh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_asinh_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_atan_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_atan_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_atan_f64(f64) -> f64
  // CHECK-LABEL: func @math_atan
  func.func @math_atan(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.atan %arg_f16 : f16
    // CHECK: llvm.call @__ocml_atan_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.atan %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.atan %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_atanh_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_atanh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_atanh_f64(f64) -> f64
  // CHECK-LABEL: func @math_atanh
  func.func @math_atanh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.atanh %arg_f16 : f16
    // CHECK: llvm.call @__ocml_atanh_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.atanh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atanh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.atanh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atanh_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_atan2_f16(f16, f16) -> f16
  // CHECK: llvm.func @__ocml_atan2_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_atan2_f64(f64, f64) -> f64
  // CHECK-LABEL: func @math_atan2
  func.func @math_atan2(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.atan2 %arg_f16, %arg_f16 : f16
    // CHECK: llvm.call @__ocml_atan2_f16(%{{.*}}, %{{.*}}) : (f16, f16) -> f16
    %result32 = math.atan2 %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan2_f32(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.atan2 %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan2_f64(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_cbrt_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_cbrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_cbrt_f64(f64) -> f64
  // CHECK-LABEL: func @math_cbrt
  func.func @math_cbrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.cbrt %arg_f16 : f16
    // CHECK: llvm.call @__ocml_cbrt_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.cbrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cbrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.cbrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cbrt_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_ceil_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_ceil_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_ceil_f64(f64) -> f64
  // CHECK-LABEL: func @math_ceil
  func.func @math_ceil(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.ceil %arg_f16 : f16
    // CHECK: llvm.call @__ocml_ceil_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.ceil %arg_f32 : f32
    // CHECK: llvm.call @__ocml_ceil_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.ceil %arg_f64 : f64
    // CHECK: llvm.call @__ocml_ceil_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_cos_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_cos_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_cos_f64(f64) -> f64
  // CHECK-LABEL: func @math_cos
  func.func @math_cos(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.cos %arg_f16 : f16
    // CHECK: llvm.call @__ocml_cos_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.cos %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cos_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.cos %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cos_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_cosh_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_cosh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_cosh_f64(f64) -> f64
  // CHECK-LABEL: func @math_cosh
  func.func @math_cosh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.cosh %arg_f16 : f16
    // CHECK: llvm.call @__ocml_cosh_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.cosh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cosh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.cosh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cosh_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_sinh_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_sinh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_sinh_f64(f64) -> f64
  // CHECK-LABEL: func @math_sinh
  func.func @math_sinh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.sinh %arg_f16 : f16
    // CHECK: llvm.call @__ocml_sinh_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.sinh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_sinh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.sinh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_sinh_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_exp_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_exp_f64(f64) -> f64
  // CHECK-LABEL: func @math_exp
  func.func @math_exp(%arg_f16 : f16, %arg_f64 : f64) -> (f16, f64) {
    %result16 = math.exp %arg_f16 : f16
    // CHECK: llvm.call @__ocml_exp_f16(%{{.*}}) : (f16) -> f16
    %result64 = math.exp %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result64 : f16, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_exp2_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_exp2_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp2_f64(f64) -> f64
  // CHECK-LABEL: func @math_exp2
  func.func @math_exp2(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.exp2 %arg_f16 : f16
    // CHECK: llvm.call @__ocml_exp2_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.exp2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp2_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.exp2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp2_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_expm1_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_expm1_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_expm1_f64(f64) -> f64
  // CHECK-LABEL: func @math_expm1
  func.func @math_expm1(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.expm1 %arg_f16 : f16
    // CHECK: llvm.call @__ocml_expm1_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.expm1 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_expm1_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.expm1 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_expm1_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_floor_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_floor_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_floor_f64(f64) -> f64
  // CHECK-LABEL: func @math_floor
  func.func @math_floor(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.floor %arg_f16 : f16
    // CHECK: llvm.call @__ocml_floor_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.floor %arg_f32 : f32
    // CHECK: llvm.call @__ocml_floor_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.floor %arg_f64 : f64
    // CHECK: llvm.call @__ocml_floor_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_log_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_log_f64(f64) -> f64
  // CHECK-LABEL: func @math_log
  func.func @math_log(%arg_f16 : f16, %arg_f64 : f64) -> (f16, f64) {
    %result16 = math.log %arg_f16 : f16
    // CHECK: llvm.call @__ocml_log_f16(%{{.*}}) : (f16) -> f16
    %result64 = math.log %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result64 : f16, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_log10_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_log10_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log10_f64(f64) -> f64
  // CHECK-LABEL: func @math_log10
  func.func @math_log10(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.log10 %arg_f16 : f16
    // CHECK: llvm.call @__ocml_log10_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.log10 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log10_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log10 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log10_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_log1p_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_log1p_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log1p_f64(f64) -> f64
  // CHECK-LABEL: func @math_log1p
  func.func @math_log1p(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.log1p %arg_f16 : f16
    // CHECK: llvm.call @__ocml_log1p_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.log1p %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log1p_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log1p %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log1p_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_pow_f16(f16, f16) -> f16
  // CHECK: llvm.func @__ocml_pow_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_pow_f64(f64, f64) -> f64
  // CHECK-LABEL: func @math_powf
  func.func @math_powf(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.powf %arg_f16, %arg_f16 : f16
    // CHECK: llvm.call @__ocml_pow_f16(%{{.*}}, %{{.*}}) : (f16, f16) -> f16
    %result32 = math.powf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_pow_f32(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.powf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_pow_f64(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_rsqrt_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_rsqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_rsqrt_f64(f64) -> f64
  // CHECK-LABEL: func @math_rsqrt
  func.func @math_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.rsqrt %arg_f16 : f16
    // CHECK: llvm.call @__ocml_rsqrt_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_rsqrt_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_sin_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_sin_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_sin_f64(f64) -> f64
  // CHECK-LABEL: func @math_sin
  func.func @math_sin(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.sin %arg_f16 : f16
    // CHECK: llvm.call @__ocml_sin_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.sin %arg_f32 : f32
    // CHECK: llvm.call @__ocml_sin_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.sin %arg_f64 : f64
    // CHECK: llvm.call @__ocml_sin_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_tanh_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_tanh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_tanh_f64(f64) -> f64
  // CHECK-LABEL: func @math_tanh
  func.func @math_tanh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.tanh %arg_f16 : f16
    // CHECK: llvm.call @__ocml_tanh_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.tanh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tanh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.tanh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tanh_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_tan_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_tan_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_tan_f64(f64) -> f64
  // CHECK-LABEL: func @math_tan
  func.func @math_tan(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.tan %arg_f16 : f16
    // CHECK: llvm.call @__ocml_tan_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.tan %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tan_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.tan %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tan_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_erf_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_erf_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_erf_f64(f64) -> f64
  // CHECK-LABEL: func @math_erf
  func.func @math_erf(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.erf %arg_f16 : f16
    // CHECK: llvm.call @__ocml_erf_f16(%{{.*}}) : (f16) -> f16
    %result32 = math.erf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_erf_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.erf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_erf_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_sin_f16(f16) -> f16
  // CHECK: llvm.func @__ocml_sin_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_sin_f64(f64) -> f64
  // CHECK-LABEL: func @math_casting
  func.func @math_casting(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64, %arg_bf16 : bf16) -> (f16, f32, f64, bf16) {
    %resultf16 = math.sin %arg_f16 : f16
    // CHECK: llvm.call @__ocml_sin_f16(%{{.*}}) : (f16) -> f16
    %resultf32 = math.sin %arg_f32 : f32
    // CHECK: llvm.call @__ocml_sin_f32(%{{.*}}) : (f32) -> f32
    %resultf64 = math.sin %arg_f64 : f64
    // CHECK: llvm.call @__ocml_sin_f64(%{{.*}}) : (f64) -> f64
    %resultbf16 = math.sin %arg_bf16 : bf16
    // CHECK: llvm.fpext %{{.*}} : bf16 to f32
    // CHECK-NEXT: llvm.call @__ocml_sin_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to bf16
    func.return %resultf16, %resultf32, %resultf64, %resultbf16 : f16, f32, f64, bf16
  }
}

// -----

module @test_module {
  // CHECK: llvm.func @__ocml_pown_f16(f16, i32) -> f16
  // CHECK: llvm.func @__ocml_pown_f32(f32, i32) -> f32
  // CHECK: llvm.func @__ocml_pown_f64(f64, i32) -> f64
  // CHECK-LABEL: func @math_fpowi
  func.func @math_fpowi(%arg0: f16, %arg1: f32, %arg2: f64, %arg3: i32) -> (f16, f32, f64) {
    // CHECK: llvm.call @__ocml_pown_f16(%{{.*}}) : (f16, i32) -> f16
    %0 = math.fpowi %arg0, %arg3 : f16, i32
    // CHECK: llvm.call @__ocml_pown_f32(%{{.*}}) : (f32, i32) -> f32
    %1 = math.fpowi %arg1, %arg3 : f32, i32
    // CHECK: llvm.call @__ocml_pown_f64(%{{.*}}) : (f64, i32) -> f64
    %2 = math.fpowi %arg2, %arg3 : f64, i32
    return %0, %1, %2 : f16, f32, f64
  }
}

// -----

// Math operation not inside function
// Ensure it not crash

module {
  "test.some_op_with_region"() ({
  ^bb0(%arg0: f64):
    // CHECK: math.atan
    %0 = math.atan %arg0 : f64
    "test.possible_terminator"() : () -> ()
  }) : () -> ()
}
