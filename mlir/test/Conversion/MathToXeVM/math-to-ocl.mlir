// RUN: mlir-opt %s -split-input-file -convert-math-to-xevm | FileCheck %s -check-prefixes='CHECK,CHECK-NO-OCL' 
// RUN: mlir-opt %s -split-input-file -convert-math-to-xevm='convert-to-ocl=true convert-arith=true' | FileCheck %s -check-prefixes='CHECK,CHECK-OCL' 

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_copysignf(f32, f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_atan2f(f32, f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_powf(f32, f32) -> f32
  // CHECK-LABEL: func @math_bin_f32
  func.func @math_bin_f32(%arg_f32_1 : f32, %arg_f32_2 : f32) -> (f32, f32, f32) {
    %result1 = math.copysign %arg_f32_1, %arg_f32_2 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_copysignf(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    // CHECK-NO-OCL: math.copysign
    %result2 = math.atan2 %arg_f32_1, %arg_f32_2 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_atan2f(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    // CHECK-NO-OCL: math.atan2
    %result3 = math.powf %arg_f32_1, %arg_f32_2 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_powf(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    // CHECK-NO-OCL: math.powf
    func.return %result1, %result2, %result3 : f32, f32, f32
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_copysignd(f64, f64) -> f64
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_atan2d(f64, f64) -> f64
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_powd(f64, f64) -> f64
  // CHECK-LABEL: func @math_bin_f64
  func.func @math_bin_f64(%arg_f64_1 : f64, %arg_f64_2 : f64) -> (f64, f64, f64) {
    %result1 = math.copysign %arg_f64_1, %arg_f64_2 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_copysignd(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    // CHECK-NO-OCL: math.copysign
    %result2 = math.atan2 %arg_f64_1, %arg_f64_2 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_atan2d(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    // CHECK-NO-OCL: math.atan2
    %result3 = math.powf %arg_f64_1, %arg_f64_2 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_powd(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    // CHECK-NO-OCL: math.powf
    func.return %result1, %result2, %result3 : f64, f64, f64
  }
}

// -----

module @test_module {
// CHECK-OCL-DAG:     llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_expm1f(f32)
// CHECK-NO-OCL-NOT:  llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_expm1f(f32)
// CHECK-LABEL: func.func @expm1_vector
  func.func @expm1_vector(%arg0: memref<32xvector<4xf32>>,
                          %arg1: memref<32xvector<4xf32>>,
                          %idx : index) {
    // CHECK: %[[ARG0:.*]] = memref.load %arg0
    %v = memref.load %arg0[%idx] : memref<32xvector<4xf32>>
    // CHECK-OCL: %[[EXT_0:.*]] = llvm.extractelement %[[ARG0]]
    // CHECK-OCL: %[[VAL_0:.*]] = llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expm1f(%[[EXT_0]])
    // CHECK-OCL: llvm.insertelement %[[VAL_0]]
    // CHECK-OCL: %[[EXT_1:.*]] = llvm.extractelement %[[ARG0]]
    // CHECK-OCL: %[[VAL_1:.*]] = llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expm1f(%[[EXT_1]])
    // CHECK-OCL: llvm.insertelement %[[VAL_1]]
    // CHECK-OCL: %[[EXT_2:.*]] = llvm.extractelement %[[ARG0]]
    // CHECK-OCL: %[[VAL_2:.*]] = llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expm1f(%[[EXT_2]])
    // CHECK-OCL: llvm.insertelement %[[VAL_2]]
    // CHECK-OCL: %[[EXT_3:.*]] = llvm.extractelement %[[ARG0]]
    // CHECK-OCL: %[[VAL_3:.*]] = llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expm1f(%[[EXT_3]])
    // CHECK-OCL: %[[INS:.*]] = llvm.insertelement %[[VAL_3]]
    // CHECK-NO-OCL: %[[INS:.*]] = math.expm1 %[[ARG0]]
    %e = math.expm1 %v : vector<4xf32>
    // CHECK: memref.store %[[INS]], %arg1
    memref.store %e, %arg1[%idx] : memref<32xvector<4xf32>>
    return
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_acosf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_acosd(f64) -> f64
  // CHECK-LABEL: func @math_acos
  func.func @math_acos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.acos %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_acosf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.acos
    %result64 = math.acos %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_acosd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.acos
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_acoshf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_acoshd(f64) -> f64
  // CHECK-LABEL: func @math_acosh
  func.func @math_acosh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.acosh %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_acoshf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.acosh
    %result64 = math.acosh %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_acoshd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.acosh
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_asinf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_asind(f64) -> f64
  // CHECK-LABEL: func @math_asin
  func.func @math_asin(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.asin %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_asinf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.asin
    %result64 = math.asin %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_asind(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.asin
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_asinhf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_asinhd(f64) -> f64
  // CHECK-LABEL: func @math_asinh
  func.func @math_asinh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.asinh %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_asinhf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.asinh
    %result64 = math.asinh %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_asinhd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.asinh
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_atanf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_atand(f64) -> f64
  // CHECK-LABEL: func @math_atan
  func.func @math_atan(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.atan %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_atanf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.atan
    %result64 = math.atan %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_atand(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.atan
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_atanhf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_atanhd(f64) -> f64
  // CHECK-LABEL: func @math_atanh
  func.func @math_atanh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.atanh %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_atanhf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.atanh
    %result64 = math.atanh %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_atanhd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.atanh
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_cbrtf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_cbrtd(f64) -> f64
  // CHECK-LABEL: func @math_cbrt
  func.func @math_cbrt(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cbrt %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_cbrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.cbrt
    %result64 = math.cbrt %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_cbrtd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.cbrt
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_cosf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_cosd(f64) -> f64
  // CHECK-LABEL: func @math_cos
  func.func @math_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cos %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_cosf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.cos
    %result64 = math.cos %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_cosd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.cos
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_coshf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_coshd(f64) -> f64
  // CHECK-LABEL: func @math_cosh
  func.func @math_cosh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cosh %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_coshf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.cosh
    %result64 = math.cosh %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_coshd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.cosh
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_erff(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_erfd(f64) -> f64
  // CHECK-LABEL: func @math_erf
  func.func @math_erf(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.erf %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_erff(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.erf
    %result64 = math.erf %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_erfd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.erf
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_erfcf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_erfcd(f64) -> f64
  // CHECK-LABEL: func @math_erfc
  func.func @math_erfc(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.erfc %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_erfcf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.erfc
    %result64 = math.erfc %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_erfcd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.erfc
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_expf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_expd(f64) -> f64
  // CHECK-LABEL: func @math_exp
  func.func @math_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.exp %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.exp
    %result64 = math.exp %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.exp
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_exp2f(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_exp2d(f64) -> f64
  // CHECK-LABEL: func @math_exp2
  func.func @math_exp2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.exp2 %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_exp2f(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.exp2
    %result64 = math.exp2 %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_exp2d(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.exp2
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_expm1f(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_expm1d(f64) -> f64
  // CHECK-LABEL: func @math_expm1
  func.func @math_expm1(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.expm1 %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expm1f(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.expm1
    %result64 = math.expm1 %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expm1d(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.expm1
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_logf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_logd(f64) -> f64
  // CHECK-LABEL: func @math_log
  func.func @math_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_logf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.log
    %result64 = math.log %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_logd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.log
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_log10f(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_log10d(f64) -> f64
  // CHECK-LABEL: func @math_log10
  func.func @math_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log10 %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_log10f(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.log10
    %result64 = math.log10 %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_log10d(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.log10
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_log1pf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_log1pd(f64) -> f64
  // CHECK-LABEL: func @math_log1p
  func.func @math_log1p(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log1p %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_log1pf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.log1p
    %result64 = math.log1p %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_log1pd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.log1p
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_log2f(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_log2d(f64) -> f64
  // CHECK-LABEL: func @math_log2
  func.func @math_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log2 %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_log2f(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.log2
    %result64 = math.log2 %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_log2d(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.log2
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_rsqrtf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_rsqrtd(f64) -> f64
  // CHECK-LABEL: func @math_rsqrt
  func.func @math_rsqrt(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.rsqrt %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_rsqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.rsqrt
    %result64 = math.rsqrt %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_rsqrtd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.rsqrt
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_sinf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_sind(f64) -> f64
  // CHECK-LABEL: func @math_sin
  func.func @math_sin(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.sin %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_sinf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.sin
    %result64 = math.sin %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_sind(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.sin
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_sinhf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_sinhd(f64) -> f64
  // CHECK-LABEL: func @math_sinh
  func.func @math_sinh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.sinh %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_sinhf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.sinh
    %result64 = math.sinh %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_sinhd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.sinh
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_sqrtf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_sqrtd(f64) -> f64
  // CHECK-LABEL: func @math_sqrt
  func.func @math_sqrt(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.sqrt %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_sqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.sqrt
    %result64 = math.sqrt %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_sqrtd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.sqrt
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_tanf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_tand(f64) -> f64
  // CHECK-LABEL: func @math_tan
  func.func @math_tan(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.tan %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_tanf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.tan
    %result64 = math.tan %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_tand(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.tan
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_tanhf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_tanhd(f64) -> f64
  // CHECK-LABEL: func @math_tanh
  func.func @math_tanh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.tanh %arg_f32 : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_tanhf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.tanh
    %result64 = math.tanh %arg_f64 : f64
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_tanhd(%{{.*}}) : (f64) -> f64
    // CHECK-NO-OCL: math.tanh
    func.return %result32, %result64 : f32, f64
  }
}

// -----

module @test_module {
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_cbrtf(f32) -> f32
  // CHECK-OCL: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_erfcf(f32) -> f32
  // CHECK-LABEL: func @math_unary_16bit
  func.func @math_unary_16bit(%arg_f16 : f16, %arg_bf16 : bf16) -> (f16, bf16) {
    %resultf16 = math.cbrt %arg_f16 : f16
    // CHECK-OCL: %[[F16:.+]] = llvm.fpext %{{.*}} : f16 to f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_cbrtf(%[[F16]]) : (f32) -> f32
    // CHECK-OCL: llvm.fptrunc %{{.*}} : f32 to f16
    // CHECK-NO-OCL: math.cbrt
    %resultbf16 = math.erfc %arg_bf16 : bf16
    // CHECK-OCL: %[[BF16:.+]] = llvm.fpext %{{.*}} : bf16 to f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_erfcf(%[[BF16]]) : (f32) -> f32
    // CHECK-OCL: llvm.fptrunc %{{.*}} : f32 to bf16
    // CHECK-NO-OCL: math.erfc
    func.return %resultf16, %resultbf16 : f16, bf16
  }
}

// -----

module @test_module {
  // CHECK-DAG: llvm.func @_Z{{.*}}__spirv_ocl_native_divideff(f32, f32) -> f32
  // CHECK-OCL-DAG: llvm.func spir_funccc @_Z{{.*}}__spirv_ocl_sqrtf(f32) -> f32
  // CHECK-LABEL: func @math_sqrt_div
  func.func @math_sqrt_div(%arg : f32) -> f32 {
    %sqrt = math.sqrt %arg : f32
    // CHECK-OCL: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_sqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NO-OCL: math.sqrt
    %result = arith.divf %arg, %sqrt fastmath<afn> : f32
    // CHECK: llvm.call @_Z{{.*}}__spirv_ocl_native_divideff(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32, f32) -> f32
    func.return %result : f32
  }
}
