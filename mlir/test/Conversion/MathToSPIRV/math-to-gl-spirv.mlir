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
  // CHECK: %[[LOG2_RECIPROCAL:.+]] = spirv.Constant 1.44269502 : f32
  // CHECK: %[[LOG0:.+]] = spirv.GL.Log {{.+}}
  // CHECK: spirv.FMul %[[LOG0]], %[[LOG2_RECIPROCAL]]
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
  // CHECK: %[[LOG2_RECIPROCAL:.+]] = spirv.Constant dense<1.44269502> : vector<3xf32>
  // CHECK: %[[LOG0:.+]] = spirv.GL.Log {{.+}}
  // CHECK: spirv.FMul %[[LOG0]], %[[LOG2_RECIPROCAL]]
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

// CHECK-LABEL: @powf_scalar
//  CHECK-SAME: (%[[LHS:.+]]: f32, %[[RHS:.+]]: f32)
func.func @powf_scalar(%lhs: f32, %rhs: f32) -> f32 {
  // CHECK: %[[F0:.+]] = spirv.Constant 0.000000e+00 : f32
  // CHECK: %[[LT:.+]] = spirv.FOrdLessThan %[[LHS]], %[[F0]] : f32
  // CHECK: %[[F1:.+]] = spirv.Constant 1.000000e+00 : f32
  // CHECK: %[[REM:.+]] = spirv.FRem %[[RHS]], %[[F1]] : f32
  // CHECK: %[[IS_FRACTION:.+]] = spirv.FOrdNotEqual %[[REM]], %[[F0]] : f32
  // CHECK: %[[AND:.+]] = spirv.LogicalAnd %[[IS_FRACTION]], %[[LT]] : i1
  // CHECK: %[[NAN:.+]] = spirv.Constant 0x7FC00000 : f32
  // CHECK: %[[NEW_LHS:.+]] = spirv.Select %[[AND]], %[[NAN]], %[[LHS]] : i1, f32
  // CHECK: %[[ABS:.+]] = spirv.GL.FAbs %[[NEW_LHS]] : f32
  // CHECK: %[[IRHS:.+]] = spirv.ConvertFToS
  // CHECK: %[[CST1:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[REM:.+]] = spirv.BitwiseAnd %[[IRHS]]
  // CHECK: %[[ODD:.+]] = spirv.IEqual %[[REM]], %[[CST1]] : i32
  // CHECK: %[[POW:.+]] = spirv.GL.Pow %[[ABS]], %[[RHS]] : f32
  // CHECK: %[[NEG:.+]] = spirv.FNegate %[[POW]] : f32
  // CHECK: %[[SNEG:.+]] = spirv.LogicalAnd %[[LT]], %[[ODD]] : i1
  // CHECK: %[[SEL:.+]] = spirv.Select %[[SNEG]], %[[NEG]], %[[POW]] : i1, f32
  %0 = math.powf %lhs, %rhs : f32
  // CHECK: return %[[SEL]]
  return %0: f32
}

// CHECK-LABEL: @powf_vector
func.func @powf_vector(%lhs: vector<4xf32>, %rhs: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spirv.FOrdLessThan
  // CHECK: spirv.FRem
  // CHECK: spirv.FOrdNotEqual
  // CHECK: spirv.LogicalAnd
  // CHECK: spirv.Select
  // CHECK: spirv.GL.FAbs
  // CHECK: spirv.BitwiseAnd %{{.*}} : vector<4xi32>
  // CHECK: spirv.IEqual %{{.*}} : vector<4xi32>
  // CHECK: spirv.GL.Pow %{{.*}}: vector<4xf32>
  // CHECK: spirv.FNegate
  // CHECK: spirv.Select
  %0 = math.powf %lhs, %rhs : vector<4xf32>
  return %0: vector<4xf32>
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
