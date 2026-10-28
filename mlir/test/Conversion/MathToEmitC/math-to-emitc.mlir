// RUN: mlir-opt -convert-math-to-emitc=language-target=c99 %s | FileCheck %s --check-prefix=c99
// RUN: mlir-opt -convert-math-to-emitc=language-target=cpp11 %s | FileCheck %s --check-prefix=cpp11
// RUN: mlir-opt -convert-to-emitc="filter-dialects=math lower-to-cpp=false" %s | FileCheck %s --check-prefix=c99
// RUN: mlir-opt -convert-to-emitc="filter-dialects=math" %s | FileCheck %s --check-prefix=cpp11

func.func @absf(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "fabsf"
    // c99-NEXT: emitc.call_opaque "fabs"
    // cpp11: emitc.call_opaque "std::fabs"
    // cpp11-NEXT: emitc.call_opaque "std::fabs"
    %0 = math.absf %arg0 : f32
    %1 = math.absf %arg1 : f64
    return
}

func.func @floor(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "floorf"
    // c99-NEXT: emitc.call_opaque "floor"
    // cpp11: emitc.call_opaque "std::floor"
    // cpp11-NEXT: emitc.call_opaque "std::floor"
    %0 = math.floor %arg0 : f32
    %1 = math.floor %arg1 : f64
    return
}

func.func @sin(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "sinf"
    // c99-NEXT: emitc.call_opaque "sin"
    // cpp11: emitc.call_opaque "std::sin"
    // cpp11-NEXT: emitc.call_opaque "std::sin"
    %0 = math.sin %arg0 : f32
    %1 = math.sin %arg1 : f64
    return
}

func.func @cos(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "cosf"
    // c99-NEXT: emitc.call_opaque "cos"
    // cpp11: emitc.call_opaque "std::cos"
    // cpp11-NEXT: emitc.call_opaque "std::cos"
    %0 = math.cos %arg0 : f32
    %1 = math.cos %arg1 : f64
    return
}

func.func @asin(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "asinf"
    // c99-NEXT: emitc.call_opaque "asin"
    // cpp11: emitc.call_opaque "std::asin"
    // cpp11-NEXT: emitc.call_opaque "std::asin"
    %0 = math.asin %arg0 : f32
    %1 = math.asin %arg1 : f64
    return
}

func.func @acos(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "acosf"
    // c99-NEXT: emitc.call_opaque "acos"
    // cpp11: emitc.call_opaque "std::acos"
    // cpp11-NEXT: emitc.call_opaque "std::acos"
    %0 = math.acos %arg0 : f32
    %1 = math.acos %arg1 : f64
    return
}

func.func @atan2(%arg0: f32, %arg1: f32, %arg2: f64, %arg3: f64) {
    // c99: emitc.call_opaque "atan2f"
    // c99-NEXT: emitc.call_opaque "atan2"
    // cpp11: emitc.call_opaque "std::atan2"
    // cpp11-NEXT: emitc.call_opaque "std::atan2"
    %0 = math.atan2 %arg0, %arg1 : f32
    %1 = math.atan2 %arg2, %arg3 : f64
    return
}

func.func @ceil(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "ceilf"
    // c99-NEXT: emitc.call_opaque "ceil"
    // cpp11: emitc.call_opaque "std::ceil"
    // cpp11-NEXT: emitc.call_opaque "std::ceil"
    %0 = math.ceil %arg0 : f32
    %1 = math.ceil %arg1 : f64
    return
}

func.func @sqrt(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "sqrtf"
    // c99-NEXT: emitc.call_opaque "sqrt"
    // cpp11: emitc.call_opaque "std::sqrt"
    // cpp11-NEXT: emitc.call_opaque "std::sqrt"
    %0 = math.sqrt %arg0 : f32
    %1 = math.sqrt %arg1 : f64
    return
}

func.func @exp(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "expf"
    // c99-NEXT: emitc.call_opaque "exp"
    // cpp11: emitc.call_opaque "std::exp"
    // cpp11-NEXT: emitc.call_opaque "std::exp"
    %0 = math.exp %arg0 : f32
    %1 = math.exp %arg1 : f64
    return
}

func.func @powf(%arg0: f32, %arg1: f32, %arg2: f64, %arg3: f64) {
    // c99: emitc.call_opaque "powf"
    // c99-NEXT: emitc.call_opaque "pow"
    // cpp11: emitc.call_opaque "std::pow"
    // cpp11-NEXT: emitc.call_opaque "std::pow"
    %0 = math.powf %arg0, %arg1 : f32
    %1 = math.powf %arg2, %arg3 : f64
    return
}

func.func @round(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "roundf"
    // c99-NEXT: emitc.call_opaque "round"
    // cpp11: emitc.call_opaque "std::round"
    // cpp11-NEXT: emitc.call_opaque "std::round"
    %0 = math.round %arg0 : f32
    %1 = math.round %arg1 : f64
    return
}

func.func @roundeven(%arg0: f32, %arg1: f64) {
  // c99: emitc.call_opaque "__builtin_roundevenf"
  // c99-NEXT: emitc.call_opaque "__builtin_roundeven"
  // cpp11: emitc.call_opaque "__builtin_roundevenf"
  // cpp11-NEXT: emitc.call_opaque "__builtin_roundeven"
  %0 = math.roundeven %arg0 : f32
  %1 = math.roundeven %arg1 : f64
  return
}

func.func @tan(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "tanf"
    // c99-NEXT: emitc.call_opaque "tan"
    // cpp11: emitc.call_opaque "std::tan"
    // cpp11-NEXT: emitc.call_opaque "std::tan"
    %0 = math.tan %arg0 : f32
    %1 = math.tan %arg1 : f64
    return
}

func.func @atan(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "atanf"
    // c99-NEXT: emitc.call_opaque "atan"
    // cpp11: emitc.call_opaque "std::atan"
    // cpp11-NEXT: emitc.call_opaque "std::atan"
    %0 = math.atan %arg0 : f32
    %1 = math.atan %arg1 : f64
    return
}

func.func @sinh(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "sinhf"
    // c99-NEXT: emitc.call_opaque "sinh"
    // cpp11: emitc.call_opaque "std::sinh"
    // cpp11-NEXT: emitc.call_opaque "std::sinh"
    %0 = math.sinh %arg0 : f32
    %1 = math.sinh %arg1 : f64
    return
}

func.func @cosh(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "coshf"
    // c99-NEXT: emitc.call_opaque "cosh"
    // cpp11: emitc.call_opaque "std::cosh"
    // cpp11-NEXT: emitc.call_opaque "std::cosh"
    %0 = math.cosh %arg0 : f32
    %1 = math.cosh %arg1 : f64
    return
}

func.func @tanh(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "tanhf"
    // c99-NEXT: emitc.call_opaque "tanh"
    // cpp11: emitc.call_opaque "std::tanh"
    // cpp11-NEXT: emitc.call_opaque "std::tanh"
    %0 = math.tanh %arg0 : f32
    %1 = math.tanh %arg1 : f64
    return
}

func.func @asinh(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "asinhf"
    // c99-NEXT: emitc.call_opaque "asinh"
    // cpp11: emitc.call_opaque "std::asinh"
    // cpp11-NEXT: emitc.call_opaque "std::asinh"
    %0 = math.asinh %arg0 : f32
    %1 = math.asinh %arg1 : f64
    return
}

func.func @acosh(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "acoshf"
    // c99-NEXT: emitc.call_opaque "acosh"
    // cpp11: emitc.call_opaque "std::acosh"
    // cpp11-NEXT: emitc.call_opaque "std::acosh"
    %0 = math.acosh %arg0 : f32
    %1 = math.acosh %arg1 : f64
    return
}

func.func @atanh(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "atanhf"
    // c99-NEXT: emitc.call_opaque "atanh"
    // cpp11: emitc.call_opaque "std::atanh"
    // cpp11-NEXT: emitc.call_opaque "std::atanh"
    %0 = math.atanh %arg0 : f32
    %1 = math.atanh %arg1 : f64
    return
}

func.func @exp2(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "exp2f"
    // c99-NEXT: emitc.call_opaque "exp2"
    // cpp11: emitc.call_opaque "std::exp2"
    // cpp11-NEXT: emitc.call_opaque "std::exp2"
    %0 = math.exp2 %arg0 : f32
    %1 = math.exp2 %arg1 : f64
    return
}

func.func @expm1(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "expm1f"
    // c99-NEXT: emitc.call_opaque "expm1"
    // cpp11: emitc.call_opaque "std::expm1"
    // cpp11-NEXT: emitc.call_opaque "std::expm1"
    %0 = math.expm1 %arg0 : f32
    %1 = math.expm1 %arg1 : f64
    return
}

func.func @log(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "logf"
    // c99-NEXT: emitc.call_opaque "log"
    // cpp11: emitc.call_opaque "std::log"
    // cpp11-NEXT: emitc.call_opaque "std::log"
    %0 = math.log %arg0 : f32
    %1 = math.log %arg1 : f64
    return
}

func.func @log2(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "log2f"
    // c99-NEXT: emitc.call_opaque "log2"
    // cpp11: emitc.call_opaque "std::log2"
    // cpp11-NEXT: emitc.call_opaque "std::log2"
    %0 = math.log2 %arg0 : f32
    %1 = math.log2 %arg1 : f64
    return
}

func.func @log10(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "log10f"
    // c99-NEXT: emitc.call_opaque "log10"
    // cpp11: emitc.call_opaque "std::log10"
    // cpp11-NEXT: emitc.call_opaque "std::log10"
    %0 = math.log10 %arg0 : f32
    %1 = math.log10 %arg1 : f64
    return
}

func.func @log1p(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "log1pf"
    // c99-NEXT: emitc.call_opaque "log1p"
    // cpp11: emitc.call_opaque "std::log1p"
    // cpp11-NEXT: emitc.call_opaque "std::log1p"
    %0 = math.log1p %arg0 : f32
    %1 = math.log1p %arg1 : f64
    return
}

func.func @cbrt(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "cbrtf"
    // c99-NEXT: emitc.call_opaque "cbrt"
    // cpp11: emitc.call_opaque "std::cbrt"
    // cpp11-NEXT: emitc.call_opaque "std::cbrt"
    %0 = math.cbrt %arg0 : f32
    %1 = math.cbrt %arg1 : f64
    return
}

func.func @erf(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "erff"
    // c99-NEXT: emitc.call_opaque "erf"
    // cpp11: emitc.call_opaque "std::erf"
    // cpp11-NEXT: emitc.call_opaque "std::erf"
    %0 = math.erf %arg0 : f32
    %1 = math.erf %arg1 : f64
    return
}

func.func @erfc(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "erfcf"
    // c99-NEXT: emitc.call_opaque "erfc"
    // cpp11: emitc.call_opaque "std::erfc"
    // cpp11-NEXT: emitc.call_opaque "std::erfc"
    %0 = math.erfc %arg0 : f32
    %1 = math.erfc %arg1 : f64
    return
}

func.func @trunc(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "truncf"
    // c99-NEXT: emitc.call_opaque "trunc"
    // cpp11: emitc.call_opaque "std::trunc"
    // cpp11-NEXT: emitc.call_opaque "std::trunc"
    %0 = math.trunc %arg0 : f32
    %1 = math.trunc %arg1 : f64
    return
}

func.func @copysign(%arg0: f32, %arg1: f32, %arg2: f64, %arg3: f64) {
    // c99: emitc.call_opaque "copysignf"(%arg0, %arg1)
    // c99-NEXT: emitc.call_opaque "copysign"(%arg2, %arg3)
    // cpp11: emitc.call_opaque "std::copysign"(%arg0, %arg1)
    // cpp11-NEXT: emitc.call_opaque "std::copysign"(%arg2, %arg3)
    %0 = math.copysign %arg0, %arg1 : f32
    %1 = math.copysign %arg2, %arg3 : f64
    return
}

func.func @fma(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f64, %arg4: f64, %arg5: f64) {
    // c99: emitc.call_opaque "fmaf"(%arg0, %arg1, %arg2)
    // c99-NEXT: emitc.call_opaque "fma"(%arg3, %arg4, %arg5)
    // cpp11: emitc.call_opaque "std::fma"(%arg0, %arg1, %arg2)
    // cpp11-NEXT: emitc.call_opaque "std::fma"(%arg3, %arg4, %arg5)
    %0 = math.fma %arg0, %arg1, %arg2 : f32
    %1 = math.fma %arg3, %arg4, %arg5 : f64
    return
}

func.func @isnan(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "isnan"(%{{.*}}) : (f32) -> i1
    // c99-NEXT: emitc.call_opaque "isnan"(%{{.*}}) : (f64) -> i1
    // cpp11: emitc.call_opaque "std::isnan"(%{{.*}}) : (f32) -> i1
    // cpp11-NEXT: emitc.call_opaque "std::isnan"(%{{.*}}) : (f64) -> i1
    %0 = math.isnan %arg0 : f32
    %1 = math.isnan %arg1 : f64
    return
}

func.func @isinf(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "isinf"(%{{.*}}) : (f32) -> i1
    // c99-NEXT: emitc.call_opaque "isinf"(%{{.*}}) : (f64) -> i1
    // cpp11: emitc.call_opaque "std::isinf"(%{{.*}}) : (f32) -> i1
    // cpp11-NEXT: emitc.call_opaque "std::isinf"(%{{.*}}) : (f64) -> i1
    %0 = math.isinf %arg0 : f32
    %1 = math.isinf %arg1 : f64
    return
}

func.func @isfinite(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "isfinite"(%{{.*}}) : (f32) -> i1
    // c99-NEXT: emitc.call_opaque "isfinite"(%{{.*}}) : (f64) -> i1
    // cpp11: emitc.call_opaque "std::isfinite"(%{{.*}}) : (f32) -> i1
    // cpp11-NEXT: emitc.call_opaque "std::isfinite"(%{{.*}}) : (f64) -> i1
    %0 = math.isfinite %arg0 : f32
    %1 = math.isfinite %arg1 : f64
    return
}

func.func @isnormal(%arg0: f32, %arg1: f64) {
    // c99: emitc.call_opaque "isnormal"(%{{.*}}) : (f32) -> i1
    // c99-NEXT: emitc.call_opaque "isnormal"(%{{.*}}) : (f64) -> i1
    // cpp11: emitc.call_opaque "std::isnormal"(%{{.*}}) : (f32) -> i1
    // cpp11-NEXT: emitc.call_opaque "std::isnormal"(%{{.*}}) : (f64) -> i1
    %0 = math.isnormal %arg0 : f32
    %1 = math.isnormal %arg1 : f64
    return
}
