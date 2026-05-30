// RUN: mlir-opt -split-input-file -convert-math-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @float32_unary_scalar
func.func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spirv.GL.Atan %{{.*}}: f32
  %0 = math.atan %arg0 : f32
  // CHECK: spirv.GL.Cos %{{.*}}: f32
  %1 = math.cos %arg0 : f32
  // CHECK: spirv.GL.Exp %{{.*}}: f32
  %2 = math.exp %arg0 : f32
  // CHECK: spirv.GL.Exp2 %{{.*}}: f32
  %exp2 = math.exp2 %arg0 : f32
  // CHECK: %[[EXP:.+]] = spirv.GL.Exp %arg0
  // CHECK: %[[ONE:.+]] = spirv.Constant 1.000000e+00 : f32
  // CHECK: spirv.FSub %[[EXP]], %[[ONE]]
  %3 = math.expm1 %arg0 : f32
  // CHECK: spirv.GL.Log %{{.*}}: f32
  %4 = math.log %arg0 : f32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1.000000e+00 : f32
  // CHECK: %[[ADDONE:.+]] = spirv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spirv.GL.Log %[[ADDONE]]
  %5 = math.log1p %arg0 : f32
  // CHECK: spirv.GL.Log2 %{{.*}}: f32
  %6 = math.log2 %arg0 : f32
  // CHECK: %[[LOG10_RECIPROCAL:.+]] = spirv.Constant 0.434294492 : f32
  // CHECK: %[[LOG1:.+]] = spirv.GL.Log {{.+}}
  // CHECK: spirv.FMul %[[LOG1]], %[[LOG10_RECIPROCAL]]
  %7 = math.log10 %arg0 : f32
  // CHECK: spirv.GL.RoundEven %{{.*}}: f32
  %8 = math.roundeven %arg0 : f32
  // CHECK: spirv.GL.InverseSqrt %{{.*}}: f32
  %9 = math.rsqrt %arg0 : f32
  // CHECK: spirv.GL.Sqrt %{{.*}}: f32
  %10 = math.sqrt %arg0 : f32
  // CHECK: spirv.GL.Tanh %{{.*}}: f32
  %11 = math.tanh %arg0 : f32
  // CHECK: spirv.GL.Sin %{{.*}}: f32
  %12 = math.sin %arg0 : f32
  // CHECK: spirv.GL.FAbs %{{.*}}: f32
  %13 = math.absf %arg0 : f32
  // CHECK: spirv.GL.Ceil %{{.*}}: f32
  %14 = math.ceil %arg0 : f32
  // CHECK: spirv.GL.Floor %{{.*}}: f32
  %15 = math.floor %arg0 : f32
  // CHECK: spirv.GL.Tan %{{.*}}: f32
  %16 = math.tan %arg0 : f32
  // CHECK: spirv.GL.Asin %{{.*}}: f32
  %17 = math.asin %arg0 : f32
  // CHECK: spirv.GL.Acos %{{.*}}: f32
  %18 = math.acos %arg0 : f32
  // CHECK: spirv.GL.Sinh %{{.*}}: f32
  %19 = math.sinh %arg0 : f32
  // CHECK: spirv.GL.Cosh %{{.*}}: f32
  %20 = math.cosh %arg0 : f32
  // CHECK: spirv.GL.Asinh %{{.*}}: f32
  %21 = math.asinh %arg0 : f32
  // CHECK: spirv.GL.Acosh %{{.*}}: f32
  %22 = math.acosh %arg0 : f32
  // CHECK: spirv.GL.Atanh %{{.*}}: f32
  %23 = math.atanh %arg0 : f32
  return
}

// CHECK-LABEL: @float32_unary_vector
func.func @float32_unary_vector(%arg0: vector<3xf32>) {
  // CHECK: spirv.GL.Atan %{{.*}}: vector<3xf32>
  %0 = math.atan %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Cos %{{.*}}: vector<3xf32>
  %1 = math.cos %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Exp %{{.*}}: vector<3xf32>
  %2 = math.exp %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Exp2 %{{.*}}: vector<3xf32>
  %exp2 = math.exp2 %arg0 : vector<3xf32>
  // CHECK: %[[EXP:.+]] = spirv.GL.Exp %arg0
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: spirv.FSub %[[EXP]], %[[ONE]]
  %3 = math.expm1 %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Log %{{.*}}: vector<3xf32>
  %4 = math.log %arg0 : vector<3xf32>
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: %[[ADDONE:.+]] = spirv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spirv.GL.Log %[[ADDONE]]
  %5 = math.log1p %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Log2 %{{.*}}: vector<3xf32>
  %6 = math.log2 %arg0 : vector<3xf32>
  // CHECK: %[[LOG10_RECIPROCAL:.+]] = spirv.Constant dense<0.434294492> : vector<3xf32>
  // CHECK: %[[LOG1:.+]] = spirv.GL.Log {{.+}}
  // CHECK: spirv.FMul %[[LOG1]], %[[LOG10_RECIPROCAL]]
  %7 = math.log10 %arg0 : vector<3xf32>
  // CHECK: spirv.GL.RoundEven %{{.*}}: vector<3xf32>
  %8 = math.roundeven %arg0 : vector<3xf32>
  // CHECK: spirv.GL.InverseSqrt %{{.*}}: vector<3xf32>
  %9 = math.rsqrt %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Sqrt %{{.*}}: vector<3xf32>
  %10 = math.sqrt %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Tanh %{{.*}}: vector<3xf32>
  %11 = math.tanh %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Sin %{{.*}}: vector<3xf32>
  %12 = math.sin %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Tan %{{.*}}: vector<3xf32>
  %13 = math.tan %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Asin %{{.*}}: vector<3xf32>
  %14 = math.asin %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Acos %{{.*}}: vector<3xf32>
  %15 = math.acos %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Sinh %{{.*}}: vector<3xf32>
  %16 = math.sinh %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Cosh %{{.*}}: vector<3xf32>
  %17 = math.cosh %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Asinh %{{.*}}: vector<3xf32>
  %18 = math.asinh %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Acosh %{{.*}}: vector<3xf32>
  %19 = math.acosh %arg0 : vector<3xf32>
  // CHECK: spirv.GL.Atanh %{{.*}}: vector<3xf32>
  %20 = math.atanh %arg0 : vector<3xf32>
  return
}

// CHECK-LABEL: @float32_ternary_scalar
func.func @float32_ternary_scalar(%a: f32, %b: f32, %c: f32) {
  // CHECK: spirv.GL.Fma %{{.*}}: f32
  %0 = math.fma %a, %b, %c : f32
  return
}

// CHECK-LABEL: @float32_ternary_vector
func.func @float32_ternary_vector(%a: vector<4xf32>, %b: vector<4xf32>,
                            %c: vector<4xf32>) {
  // CHECK: spirv.GL.Fma %{{.*}}: vector<4xf32>
  %0 = math.fma %a, %b, %c : vector<4xf32>
  return
}

// CHECK-LABEL: @float32_clamp_scalar
func.func @float32_clamp_scalar(%value: f32, %min: f32, %max: f32) {
  // CHECK: spirv.GL.FClamp %{{.*}}, %{{.*}}, %{{.*}} : f32
  %0 = math.clampf %value to [%min, %max] : f32
  return
}

// CHECK-LABEL: @float32_clamp_vector
func.func @float32_clamp_vector(%value: vector<4xf32>, %min: vector<4xf32>,
                                %max: vector<4xf32>) {
  // CHECK: spirv.GL.FClamp %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
  %0 = math.clampf %value to [%min, %max] : vector<4xf32>
  return
}

// CHECK-LABEL: @int_unary
func.func @int_unary(%arg0: i32) {
  // CHECK: spirv.GL.SAbs %{{.*}}
  %0 = math.absi %arg0 : i32
  return
}

// CHECK-LABEL: @ctlz_scalar
//  CHECK-SAME: (%[[VAL:.+]]: i32)
func.func @ctlz_scalar(%val: i32) -> i32 {
  // CHECK-DAG: %[[V1:.+]] = spirv.Constant 1 : i32
  // CHECK-DAG: %[[V31:.+]] = spirv.Constant 31 : i32
  // CHECK-DAG: %[[V32:.+]] = spirv.Constant 32 : i32
  // CHECK: %[[MSB:.+]] = spirv.GL.FindUMsb %[[VAL]] : i32
  // CHECK: %[[SUB1:.+]] = spirv.ISub %[[V31]], %[[MSB]] : i32
  // CHECK: %[[SUB2:.+]] = spirv.ISub %[[V32]], %[[VAL]] : i32
  // CHECK: %[[CMP:.+]] = spirv.ULessThanEqual %[[VAL]], %[[V1]] : i32
  // CHECK: %[[R:.+]] = spirv.Select %[[CMP]], %[[SUB2]], %[[SUB1]] : i1, i32
  // CHECK: return %[[R]]
  %0 = math.ctlz %val : i32
  return %0 : i32
}

// CHECK-LABEL: @ctlz_vector1
func.func @ctlz_vector1(%val: vector<1xi32>) -> vector<1xi32> {
  // CHECK: spirv.GL.FindUMsb
  // CHECK: spirv.ISub
  // CHECK: spirv.ULessThanEqual
  // CHECK: spirv.Select
  %0 = math.ctlz %val : vector<1xi32>
  return %0 : vector<1xi32>
}

// CHECK-LABEL: @ctlz_vector2
//  CHECK-SAME: (%[[VAL:.+]]: vector<2xi32>)
func.func @ctlz_vector2(%val: vector<2xi32>) -> vector<2xi32> {
  // CHECK-DAG: %[[V1:.+]] = spirv.Constant dense<1> : vector<2xi32>
  // CHECK-DAG: %[[V31:.+]] = spirv.Constant dense<31> : vector<2xi32>
  // CHECK-DAG: %[[V32:.+]] = spirv.Constant dense<32> : vector<2xi32>
  // CHECK: %[[MSB:.+]] = spirv.GL.FindUMsb %[[VAL]] : vector<2xi32>
  // CHECK: %[[SUB1:.+]] = spirv.ISub %[[V31]], %[[MSB]] : vector<2xi32>
  // CHECK: %[[SUB2:.+]] = spirv.ISub %[[V32]], %[[VAL]] : vector<2xi32>
  // CHECK: %[[CMP:.+]] = spirv.ULessThanEqual %[[VAL]], %[[V1]] : vector<2xi32>
  // CHECK: %[[R:.+]] = spirv.Select %[[CMP]], %[[SUB2]], %[[SUB1]] : vector<2xi1>, vector<2xi32>
  %0 = math.ctlz %val : vector<2xi32>
  return %0 : vector<2xi32>
}

// Dynamic exponent: exp(y * log(x)); yields NaN for x<0.
// CHECK-LABEL: @powf_scalar
//  CHECK-SAME: (%[[LHS:.+]]: f32, %[[RHS:.+]]: f32)
func.func @powf_scalar(%lhs: f32, %rhs: f32) -> f32 {
  // CHECK: %[[LOG:.+]] = spirv.GL.Log %[[LHS]] : f32
  // CHECK: %[[MUL:.+]] = spirv.FMul %[[RHS]], %[[LOG]] : f32
  // CHECK: %[[EXP:.+]] = spirv.GL.Exp %[[MUL]] : f32
  %0 = math.powf %lhs, %rhs : f32
  // CHECK: return %[[EXP]]
  return %0: f32
}

// CHECK-LABEL: @powf_vector
func.func @powf_vector(%lhs: vector<4xf32>, %rhs: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spirv.GL.Log %{{.*}} : vector<4xf32>
  // CHECK: spirv.FMul %{{.*}} : vector<4xf32>
  // CHECK: spirv.GL.Exp %{{.*}} : vector<4xf32>
  %0 = math.powf %lhs, %rhs : vector<4xf32>
  return %0: vector<4xf32>
}

// Constant odd integer exponent: parity is known statically, so the lowering
// drops the runtime FToS/BitwiseAnd/IEqual/LogicalAnd parity computation.
// CHECK-LABEL: @powf_const_odd_int_exp
//  CHECK-SAME: (%[[LHS:.+]]: f32)
func.func @powf_const_odd_int_exp(%lhs: f32) -> f32 {
  // CHECK: %[[RHS:.+]] = arith.constant 3.000000e+00 : f32
  // CHECK: %[[ABS:.+]] = spirv.GL.FAbs %[[LHS]] : f32
  // CHECK: %[[POW:.+]] = spirv.GL.Pow %[[ABS]], %[[RHS]] : f32
  // CHECK: %[[F0:.+]] = spirv.Constant 0.000000e+00 : f32
  // CHECK: %[[LT:.+]] = spirv.FOrdLessThan %[[LHS]], %[[F0]] : f32
  // CHECK: %[[NEG:.+]] = spirv.FNegate %[[POW]] : f32
  // CHECK: %[[SEL:.+]] = spirv.Select %[[LT]], %[[NEG]], %[[POW]] : i1, f32
  %c = arith.constant 3.0 : f32
  %0 = math.powf %lhs, %c : f32
  // CHECK: return %[[SEL]]
  return %0: f32
}

// Constant even integer exponent: result is non-negative, no select needed.
// CHECK-LABEL: @powf_const_even_int_exp
//  CHECK-SAME: (%[[LHS:.+]]: f32)
func.func @powf_const_even_int_exp(%lhs: f32) -> f32 {
  // CHECK: %[[RHS:.+]] = arith.constant 4.000000e+00 : f32
  // CHECK: %[[ABS:.+]] = spirv.GL.FAbs %[[LHS]] : f32
  // CHECK: %[[POW:.+]] = spirv.GL.Pow %[[ABS]], %[[RHS]] : f32
  %c = arith.constant 4.0 : f32
  %0 = math.powf %lhs, %c : f32
  // CHECK: return %[[POW]]
  return %0: f32
}

// Constant non-integer exponent: falls into the dynamic exp(y*log(x)) path.
// CHECK-LABEL: @powf_const_frac_exp
//  CHECK-SAME: (%[[LHS:.+]]: f32)
func.func @powf_const_frac_exp(%lhs: f32) -> f32 {
  // CHECK: %[[RHS:.+]] = arith.constant 2.500000e+00 : f32
  // CHECK: %[[LOG:.+]] = spirv.GL.Log %[[LHS]] : f32
  // CHECK: %[[MUL:.+]] = spirv.FMul %[[RHS]], %[[LOG]] : f32
  // CHECK: %[[EXP:.+]] = spirv.GL.Exp %[[MUL]] : f32
  %c = arith.constant 2.5 : f32
  %0 = math.powf %lhs, %c : f32
  // CHECK: return %[[EXP]]
  return %0: f32
}

// Splat constant odd integer-valued vector exponent: uniform odd parity.
// CHECK-LABEL: @powf_const_odd_int_exp_vector
func.func @powf_const_odd_int_exp_vector(%lhs: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spirv.GL.FAbs
  // CHECK: spirv.GL.Pow %{{.*}}: vector<4xf32>
  // CHECK: spirv.FOrdLessThan
  // CHECK: spirv.FNegate
  // CHECK: spirv.Select
  %c = arith.constant dense<3.0> : vector<4xf32>
  %0 = math.powf %lhs, %c : vector<4xf32>
  return %0: vector<4xf32>
}

// Mixed-parity constant integer-valued vector exponent: per-element odd-mask
// constant is materialized and AND-ed with lhs<0.
// CHECK-LABEL: @powf_const_mixed_int_exp_vector
func.func @powf_const_mixed_int_exp_vector(%lhs: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spirv.GL.FAbs
  // CHECK: spirv.GL.Pow %{{.*}}: vector<4xf32>
  // CHECK: spirv.FOrdLessThan
  // CHECK: spirv.FNegate
  // CHECK: %[[ODD:.+]] = spirv.Constant dense<[true, false, true, false]> : vector<4xi1>
  // CHECK: spirv.LogicalAnd %{{.*}}, %[[ODD]] : vector<4xi1>
  // CHECK: spirv.Select
  %c = arith.constant dense<[3.0, 2.0, 5.0, 4.0]> : vector<4xf32>
  %0 = math.powf %lhs, %c : vector<4xf32>
  return %0: vector<4xf32>
}

// CHECK-LABEL: @fpowi_scalar
//  CHECK-SAME: (%[[BASE:.+]]: f32, %[[POW:.+]]: i32)
func.func @fpowi_scalar(%base: f32, %power: i32) -> f32 {
  // CHECK: %[[EXP:.+]] = spirv.ConvertSToF %[[POW]] : i32 to f32
  // CHECK: %[[ABS:.+]] = spirv.GL.FAbs %[[BASE]] : f32
  // CHECK: %[[POWF:.+]] = spirv.GL.Pow %[[ABS]], %[[EXP]] : f32
  // CHECK: %[[F0:.+]] = spirv.Constant 0.000000e+00 : f32
  // CHECK: %[[LT:.+]] = spirv.FOrdLessThan %[[BASE]], %[[F0]] : f32
  // CHECK: %[[I1:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[AND:.+]] = spirv.BitwiseAnd %[[POW]], %[[I1]] : i32
  // CHECK: %[[ODD:.+]] = spirv.IEqual %[[AND]], %[[I1]] : i32
  // CHECK: %[[NEG_C:.+]] = spirv.LogicalAnd %[[LT]], %[[ODD]] : i1
  // CHECK: %[[NEG:.+]] = spirv.FNegate %[[POWF]] : f32
  // CHECK: %[[SEL:.+]] = spirv.Select %[[NEG_C]], %[[NEG]], %[[POWF]] : i1, f32
  %0 = math.fpowi %base, %power : f32, i32
  // CHECK: return %[[SEL]]
  return %0 : f32
}

// CHECK-LABEL: @fpowi_vector
func.func @fpowi_vector(%base: vector<4xf32>, %power: vector<4xi32>) -> vector<4xf32> {
  // CHECK: spirv.ConvertSToF %{{.*}} : vector<4xi32> to vector<4xf32>
  // CHECK: spirv.GL.FAbs %{{.*}} : vector<4xf32>
  // CHECK: spirv.GL.Pow %{{.*}} : vector<4xf32>
  // CHECK: spirv.FOrdLessThan %{{.*}} : vector<4xf32>
  // CHECK: spirv.BitwiseAnd %{{.*}} : vector<4xi32>
  // CHECK: spirv.IEqual %{{.*}} : vector<4xi32>
  // CHECK: spirv.LogicalAnd %{{.*}} : vector<4xi1>
  // CHECK: spirv.FNegate %{{.*}} : vector<4xf32>
  // CHECK: spirv.Select %{{.*}} : vector<4xi1>, vector<4xf32>
  %0 = math.fpowi %base, %power : vector<4xf32>, vector<4xi32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @round_scalar
func.func @round_scalar(%x: f32) -> f32 {
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0.000000e+00
  // CHECK: %[[ONE:.+]] = spirv.Constant 1.000000e+00
  // CHECK: %[[HALF:.+]] = spirv.Constant 5.000000e-01
  // CHECK: %[[ABS:.+]] = spirv.GL.FAbs %arg0
  // CHECK: %[[FLOOR:.+]] = spirv.GL.Floor %[[ABS]]
  // CHECK: %[[SUB:.+]] = spirv.FSub %[[ABS]], %[[FLOOR]]
  // CHECK: %[[GE:.+]] = spirv.FOrdGreaterThanEqual %[[SUB]], %[[HALF]]
  // CHECK: %[[SEL:.+]] = spirv.Select %[[GE]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[ADD:.+]] = spirv.FAdd %[[FLOOR]], %[[SEL]]
  // CHECK: %[[BITCAST:.+]] = spirv.Bitcast %[[ADD]]
  %0 = math.round %x : f32
  return %0: f32
}

// CHECK-LABEL: @round_vector
func.func @round_vector(%x: vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[ZERO:.+]] = spirv.Constant dense<0.000000e+00>
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1.000000e+00>
  // CHECK: %[[HALF:.+]] = spirv.Constant dense<5.000000e-01>
  // CHECK: %[[ABS:.+]] = spirv.GL.FAbs %arg0
  // CHECK: %[[FLOOR:.+]] = spirv.GL.Floor %[[ABS]]
  // CHECK: %[[SUB:.+]] = spirv.FSub %[[ABS]], %[[FLOOR]]
  // CHECK: %[[GE:.+]] = spirv.FOrdGreaterThanEqual %[[SUB]], %[[HALF]]
  // CHECK: %[[SEL:.+]] = spirv.Select %[[GE]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[ADD:.+]] = spirv.FAdd %[[FLOOR]], %[[SEL]]
  // CHECK: %[[BITCAST:.+]] = spirv.Bitcast %[[ADD]]
  %0 = math.round %x : vector<4xf32>
  return %0: vector<4xf32>
}

// Unit dimensional vectors are converted to scalars by inserting
// unrealized_conversion_cast's.
//
// CHECK-LABEL: @round_vector_unit_dim
//  CHECK-SAME: (%[[ARG:.+]]: vector<1xf32>) -> vector<1xf32>
func.func @round_vector_unit_dim(%x: vector<1xf32>) -> vector<1xf32> {
  // CHECK: %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<1xf32> to f32
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0.000000e+00
  // CHECK: %[[ONE:.+]] = spirv.Constant 1.000000e+00
  // CHECK: %[[HALF:.+]] = spirv.Constant 5.000000e-01
  // CHECK: %[[ABS:.+]] = spirv.GL.FAbs %[[CAST]] : f32
  // CHECK: %[[FLOOR:.+]] = spirv.GL.Floor %[[ABS]]
  // CHECK: %[[SUB:.+]] = spirv.FSub %[[ABS]], %[[FLOOR]]
  // CHECK: %[[GE:.+]] = spirv.FOrdGreaterThanEqual %[[SUB]], %[[HALF]]
  // CHECK: %[[SEL:.+]] = spirv.Select %[[GE]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[ADD:.+]] = spirv.FAdd %[[FLOOR]], %[[SEL]]
  // CHECK: %[[BITCAST:.+]] = spirv.Bitcast %[[ADD]] : f32 to i32
  %0 = math.round %x : vector<1xf32>
  return %0: vector<1xf32>
}

} // end module

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64, Int16], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @ctlz_scalar
func.func @ctlz_scalar(%val: i64) -> i64 {
  // CHECK: math.ctlz
  %0 = math.ctlz %val : i64
  return %0 : i64
}

// CHECK-LABEL: @ctlz_vector2
func.func @ctlz_vector2(%val: vector<2xi16>) -> vector<2xi16> {
  // CHECK: math.ctlz
  %0 = math.ctlz %val : vector<2xi16>
  return %0 : vector<2xi16>
}

} // end module

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {

// 2-D vectors are not supported.

// CHECK-LABEL: @vector_2d
func.func @vector_2d(%arg0: vector<2x2xf32>) {
  // CHECK-NEXT: math.atan {{.+}} : vector<2x2xf32>
  %0 = math.atan %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.cos {{.+}} : vector<2x2xf32>
  %1 = math.cos %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.exp {{.+}} : vector<2x2xf32>
  %2 = math.exp %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.absf {{.+}} : vector<2x2xf32>
  %3 = math.absf %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.ceil {{.+}} : vector<2x2xf32>
  %4 = math.ceil %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.floor {{.+}} : vector<2x2xf32>
  %5 = math.floor %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.powf {{.+}}, {{%.+}} : vector<2x2xf32>
  %6 = math.powf %arg0, %arg0 : vector<2x2xf32>
  // CHECK-NEXT: return
  return
}

// Tensors are not supported.

// CHECK-LABEL: @tensor_1d
func.func @tensor_1d(%arg0: tensor<2xf32>) {
  // CHECK-NEXT: math.atan {{.+}} : tensor<2xf32>
  %0 = math.atan %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.cos {{.+}} : tensor<2xf32>
  %1 = math.cos %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.exp {{.+}} : tensor<2xf32>
  %2 = math.exp %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.absf {{.+}} : tensor<2xf32>
  %3 = math.absf %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.ceil {{.+}} : tensor<2xf32>
  %4 = math.ceil %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.floor {{.+}} : tensor<2xf32>
  %5 = math.floor %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.powf {{.+}}, {{%.+}} : tensor<2xf32>
  %6 = math.powf %arg0, %arg0 : tensor<2xf32>
  // CHECK-NEXT: return
  return
}

} // end module
