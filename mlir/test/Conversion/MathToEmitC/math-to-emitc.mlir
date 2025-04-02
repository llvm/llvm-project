// RUN: mlir-opt -convert-math-to-emitc=language-target=c99 %s | FileCheck %s --check-prefix=c99
// RUN: mlir-opt -convert-math-to-emitc=language-target=cpp11 %s | FileCheck %s --check-prefix=cpp11

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
