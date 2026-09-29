// RUN: mlir-opt %s -convert-math-to-xevm='convert-to-ocl=false' \
// RUN:   | FileCheck %s -check-prefixes='CHECK,CHECK-ARITH'
// RUN: mlir-opt %s -convert-math-to-xevm='convert-to-ocl=false convert-arith=false' \
// RUN:   | FileCheck %s -check-prefixes='CHECK,CHECK-NO-ARITH'

// Check that MathToXeVM simplifies math ops before lowering them, so the
// cheaper form is what reaches the intrinsics. Every op here is marked `afn`,
// so this test turns convert-to-ocl off and pins the native intrinsics.

// CHECK-LABEL: func @powf_strength_reduction
// CHECK-SAME: (%[[X:.*]]: f32)
// An exponent of 1, 2 or 3 needs no power call at all:
// CHECK-NOT: native_powr
func.func @powf_strength_reduction(%x: f32) -> (f32, f32, f32, f32, f32) {
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %c0_5 = arith.constant 0.5 : f32
  %cm0_5 = arith.constant -0.5 : f32

  // pow(x, 1) is x itself.
  %pow1 = math.powf %x, %c1 fastmath<afn> : f32

  // CHECK: %[[SQUARE:.*]] = arith.mulf %[[X]], %[[X]] fastmath<afn> : f32
  %pow2 = math.powf %x, %c2 fastmath<afn> : f32

  // CHECK: %[[CUBE_SQUARE:.*]] = arith.mulf %[[X]], %[[X]] fastmath<afn> : f32
  // CHECK: %[[CUBE:.*]] = arith.mulf %[[X]], %[[CUBE_SQUARE]] fastmath<afn> : f32
  %pow3 = math.powf %x, %c3 fastmath<afn> : f32

  // A halved exponent becomes a square root, which is lowered as usual:
  // CHECK: %[[SQRT:.*]] = llvm.call @_Z23__spirv_ocl_native_sqrtf(%[[X]]) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
  %pow0_5 = math.powf %x, %c0_5 fastmath<afn> : f32

  // CHECK: %[[RSQRT:.*]] = llvm.call @_Z24__spirv_ocl_native_rsqrtf(%[[X]]) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
  %powm0_5 = math.powf %x, %cm0_5 fastmath<afn> : f32

  // CHECK: return %[[X]], %[[SQUARE]], %[[CUBE]], %[[SQRT]], %[[RSQRT]]
  return %pow1, %pow2, %pow3, %pow0_5, %powm0_5 : f32, f32, f32, f32, f32
}

// `exp(a) / exp(b)` becomes a single `exp(a - b)`. The fold has to happen
// before the lowering: once the exponentials are calls they are no longer dead,
// so all three would remain.

// CHECK-LABEL: func @exp_quotient
// CHECK-SAME: (%[[A:.*]]: f32, %[[B:.*]]: f32)
// CHECK-NOT: native_expf
func.func @exp_quotient(%a: f32, %b: f32) -> f32 {
  // CHECK: %[[DIFF:.*]] = arith.subf %[[A]], %[[B]]
  // CHECK: %[[EXP:.*]] = llvm.call @_Z22__spirv_ocl_native_expf(%[[DIFF]])
  %exp_a = math.exp %a fastmath<fast> : f32
  %exp_b = math.exp %b fastmath<fast> : f32
  %quotient = arith.divf %exp_a, %exp_b fastmath<fast> : f32
  // Only one exponential is left, and the division is gone:
  // CHECK-NOT: native_expf
  // CHECK-NOT: native_divide
  // CHECK: return %[[EXP]]
  return %quotient : f32
}

// The fold needs `arcp` or `reassoc`: `afn` alone is not enough to reassociate
// the division, so all three operations stay.

// CHECK-LABEL: func @exp_quotient_afn_only
func.func @exp_quotient_afn_only(%a: f32, %b: f32) -> f32 {
  // CHECK: %[[EXP_A:.*]] = llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}})
  // CHECK: %[[EXP_B:.*]] = llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}})
  %exp_a = math.exp %a fastmath<afn> : f32
  %exp_b = math.exp %b fastmath<afn> : f32
  // CHECK-ARITH: llvm.call @_Z25__spirv_ocl_native_divideff(%[[EXP_A]], %[[EXP_B]])
  // CHECK-NO-ARITH: arith.divf %[[EXP_A]], %[[EXP_B]]
  %quotient = arith.divf %exp_a, %exp_b fastmath<afn> : f32
  return %quotient : f32
}

// Ops the simplifications cannot rewrite are left untouched: a non-constant or
// unhandled exponent still goes to the native power intrinsic.

// CHECK-LABEL: func @powf_not_simplified
func.func @powf_not_simplified(%x: f32, %y: f32) -> (f32, f32) {
  %c4 = arith.constant 4.0 : f32
  // CHECK: llvm.call @_Z23__spirv_ocl_native_powrff(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32, f32) -> f32
  %pow_var = math.powf %x, %y fastmath<afn> : f32
  // CHECK: llvm.call @_Z23__spirv_ocl_native_powrff(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32, f32) -> f32
  %pow4 = math.powf %x, %c4 fastmath<afn> : f32
  return %pow_var, %pow4 : f32, f32
}
