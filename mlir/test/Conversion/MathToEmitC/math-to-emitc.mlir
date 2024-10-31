// RUN: mlir-opt -convert-math-to-emitc=language-target=C %s | FileCheck %s --check-prefix=C
// RUN: mlir-opt -convert-math-to-emitc=language-target=CPP %s | FileCheck %s --check-prefix=CPP

func.func @absf_to_call_opaque(%arg0: f32) {
    // C: emitc.call_opaque "fabsf"
    // CPP: emitc.call_opaque "std::fabs"
    %1 = math.absf %arg0 : f32
    return
  }
func.func @floor_to_call_opaque(%arg0: f32) {
    // C: emitc.call_opaque "floorf"
    // CPP: emitc.call_opaque "std::floor"
    %1 = math.floor %arg0 : f32
    return
  }
func.func @sin_to_call_opaque(%arg0: f32) {
    // C: emitc.call_opaque "sinf"
    // CPP: emitc.call_opaque "std::sin"  
    %1 = math.sin %arg0 : f32
    return
  }
func.func @cos_to_call_opaque(%arg0: f32) {
    // C: emitc.call_opaque "cosf"
    // CPP: emitc.call_opaque "std::cos"    
    %1 = math.cos %arg0 : f32
    return
  }
func.func @asin_to_call_opaque(%arg0: f32) {
    // C: emitc.call_opaque "asinf"
    // CPP: emitc.call_opaque "std::asin"     
    %1 = math.asin %arg0 : f32
    return
  }
func.func @acos_to_call_opaque(%arg0: f64) {
    // C: emitc.call_opaque "acos"
    // CPP: emitc.call_opaque "std::acos"      
    %1 = math.acos %arg0 : f64
    return
  }
func.func @atan2_to_call_opaque(%arg0: f64, %arg1: f64) {
    // C: emitc.call_opaque "atan2"
    // CPP: emitc.call_opaque "std::atan2"       
    %1 = math.atan2 %arg0, %arg1 : f64
    return
  }
func.func @ceil_to_call_opaque(%arg0: f64) {
    // C: emitc.call_opaque "ceil"
    // CPP: emitc.call_opaque "std::ceil"    
    %1 = math.ceil %arg0 : f64
    return
  }
func.func @exp_to_call_opaque(%arg0: f64) {
    // C: emitc.call_opaque "exp"
    // CPP: emitc.call_opaque "std::exp"    
    %1 = math.exp %arg0 : f64
    return
  }
func.func @powf_to_call_opaque(%arg0: f64, %arg1: f64) {
    // C: emitc.call_opaque "pow"
    // CPP: emitc.call_opaque "std::pow"    
    %1 = math.powf %arg0, %arg1 : f64
    return
  }


