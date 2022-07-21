// RUN: mlir-opt -split-input-file -convert-math-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], []>, #spv.resource_limits<>>
} {

// CHECK-LABEL: @float32_unary_scalar
func.func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spv.GL.Cos %{{.*}}: f32
  %0 = math.cos %arg0 : f32
  // CHECK: spv.GL.Exp %{{.*}}: f32
  %1 = math.exp %arg0 : f32
  // CHECK: %[[EXP:.+]] = spv.GL.Exp %arg0
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f32
  // CHECK: spv.FSub %[[EXP]], %[[ONE]]
  %2 = math.expm1 %arg0 : f32
  // CHECK: spv.GL.Log %{{.*}}: f32
  %3 = math.log %arg0 : f32
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f32
  // CHECK: %[[ADDONE:.+]] = spv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spv.GL.Log %[[ADDONE]]
  %4 = math.log1p %arg0 : f32
  // CHECK: spv.GL.InverseSqrt %{{.*}}: f32
  %5 = math.rsqrt %arg0 : f32
  // CHECK: spv.GL.Sqrt %{{.*}}: f32
  %6 = math.sqrt %arg0 : f32
  // CHECK: spv.GL.Tanh %{{.*}}: f32
  %7 = math.tanh %arg0 : f32
  // CHECK: spv.GL.Sin %{{.*}}: f32
  %8 = math.sin %arg0 : f32
  // CHECK: spv.GL.FAbs %{{.*}}: f32
  %9 = math.abs %arg0 : f32
  // CHECK: spv.GL.Ceil %{{.*}}: f32
  %10 = math.ceil %arg0 : f32
  // CHECK: spv.GL.Floor %{{.*}}: f32
  %11 = math.floor %arg0 : f32
  return
}

// CHECK-LABEL: @float32_unary_vector
func.func @float32_unary_vector(%arg0: vector<3xf32>) {
  // CHECK: spv.GL.Cos %{{.*}}: vector<3xf32>
  %0 = math.cos %arg0 : vector<3xf32>
  // CHECK: spv.GL.Exp %{{.*}}: vector<3xf32>
  %1 = math.exp %arg0 : vector<3xf32>
  // CHECK: %[[EXP:.+]] = spv.GL.Exp %arg0
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: spv.FSub %[[EXP]], %[[ONE]]
  %2 = math.expm1 %arg0 : vector<3xf32>
  // CHECK: spv.GL.Log %{{.*}}: vector<3xf32>
  %3 = math.log %arg0 : vector<3xf32>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: %[[ADDONE:.+]] = spv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spv.GL.Log %[[ADDONE]]
  %4 = math.log1p %arg0 : vector<3xf32>
  // CHECK: spv.GL.InverseSqrt %{{.*}}: vector<3xf32>
  %5 = math.rsqrt %arg0 : vector<3xf32>
  // CHECK: spv.GL.Sqrt %{{.*}}: vector<3xf32>
  %6 = math.sqrt %arg0 : vector<3xf32>
  // CHECK: spv.GL.Tanh %{{.*}}: vector<3xf32>
  %7 = math.tanh %arg0 : vector<3xf32>
  // CHECK: spv.GL.Sin %{{.*}}: vector<3xf32>
  %8 = math.sin %arg0 : vector<3xf32>
  return
}

// CHECK-LABEL: @float32_ternary_scalar
func.func @float32_ternary_scalar(%a: f32, %b: f32, %c: f32) {
  // CHECK: spv.GL.Fma %{{.*}}: f32
  %0 = math.fma %a, %b, %c : f32
  return
}

// CHECK-LABEL: @float32_ternary_vector
func.func @float32_ternary_vector(%a: vector<4xf32>, %b: vector<4xf32>,
                            %c: vector<4xf32>) {
  // CHECK: spv.GL.Fma %{{.*}}: vector<4xf32>
  %0 = math.fma %a, %b, %c : vector<4xf32>
  return
}

// CHECK-LABEL: @ctlz_scalar
//  CHECK-SAME: (%[[VAL:.+]]: i32)
func.func @ctlz_scalar(%val: i32) -> i32 {
  // CHECK-DAG: %[[V1:.+]] = spv.Constant 1 : i32
  // CHECK-DAG: %[[V31:.+]] = spv.Constant 31 : i32
  // CHECK-DAG: %[[V32:.+]] = spv.Constant 32 : i32
  // CHECK: %[[MSB:.+]] = spv.GL.FindUMsb %[[VAL]] : i32
  // CHECK: %[[SUB1:.+]] = spv.ISub %[[V31]], %[[MSB]] : i32
  // CHECK: %[[SUB2:.+]] = spv.ISub %[[V32]], %[[VAL]] : i32
  // CHECK: %[[CMP:.+]] = spv.ULessThanEqual %[[VAL]], %[[V1]] : i32
  // CHECK: %[[R:.+]] = spv.Select %[[CMP]], %[[SUB2]], %[[SUB1]] : i1, i32
  // CHECK: return %[[R]]
  %0 = math.ctlz %val : i32
  return %0 : i32
}

// CHECK-LABEL: @ctlz_vector1
func.func @ctlz_vector1(%val: vector<1xi32>) -> vector<1xi32> {
  // CHECK: spv.GL.FindUMsb
  // CHECK: spv.ISub
  // CHECK: spv.ULessThanEqual
  // CHECK: spv.Select
  %0 = math.ctlz %val : vector<1xi32>
  return %0 : vector<1xi32>
}

// CHECK-LABEL: @ctlz_vector2
//  CHECK-SAME: (%[[VAL:.+]]: vector<2xi32>)
func.func @ctlz_vector2(%val: vector<2xi32>) -> vector<2xi32> {
  // CHECK-DAG: %[[V1:.+]] = spv.Constant dense<1> : vector<2xi32>
  // CHECK-DAG: %[[V31:.+]] = spv.Constant dense<31> : vector<2xi32>
  // CHECK-DAG: %[[V32:.+]] = spv.Constant dense<32> : vector<2xi32>
  // CHECK: %[[MSB:.+]] = spv.GL.FindUMsb %[[VAL]] : vector<2xi32>
  // CHECK: %[[SUB1:.+]] = spv.ISub %[[V31]], %[[MSB]] : vector<2xi32>
  // CHECK: %[[SUB2:.+]] = spv.ISub %[[V32]], %[[VAL]] : vector<2xi32>
  // CHECK: %[[CMP:.+]] = spv.ULessThanEqual %[[VAL]], %[[V1]] : vector<2xi32>
  // CHECK: %[[R:.+]] = spv.Select %[[CMP]], %[[SUB2]], %[[SUB1]] : vector<2xi1>, vector<2xi32>
  %0 = math.ctlz %val : vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL: @powf_scalar
//  CHECK-SAME: (%[[LHS:.+]]: f32, %[[RHS:.+]]: f32)
func.func @powf_scalar(%lhs: f32, %rhs: f32) -> f32 {
  // CHECK: %[[F0:.+]] = spv.Constant 0.000000e+00 : f32
  // CHECK: %[[LT:.+]] = spv.FOrdLessThan %[[LHS]], %[[F0]] : f32
  // CHECK: %[[ABS:.+]] = spv.GL.FAbs %[[LHS]] : f32
  // CHECK: %[[POW:.+]] = spv.GL.Pow %[[ABS]], %[[RHS]] : f32
  // CHECK: %[[NEG:.+]] = spv.FNegate %[[POW]] : f32
  // CHECK: %[[SEL:.+]] = spv.Select %[[LT]], %[[NEG]], %[[POW]] : i1, f32
  %0 = math.powf %lhs, %rhs : f32
  // CHECK: return %[[SEL]]
  return %0: f32
}

// CHECK-LABEL: @powf_vector
func.func @powf_vector(%lhs: vector<4xf32>, %rhs: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spv.FOrdLessThan
  // CHEKC: spv.GL.FAbs
  // CHECK: spv.GL.Pow %{{.*}}: vector<4xf32>
  // CHECK: spv.FNegate
  // CHECK: spv.Select
  %0 = math.powf %lhs, %rhs : vector<4xf32>
  return %0: vector<4xf32>
}

// CHECK-LABEL: @round_scalar
func.func @round_scalar(%x: f32) -> f32 {
  // CHECK: %[[ZERO:.+]] = spv.Constant 0.000000e+00
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00
  // CHECK: %[[HALF:.+]] = spv.Constant 5.000000e-01
  // CHECK: %[[ABS:.+]] = spv.GL.FAbs %arg0
  // CHECK: %[[FLOOR:.+]] = spv.GL.Floor %[[ABS]]
  // CHECK: %[[SUB:.+]] = spv.FSub %[[ABS]], %[[FLOOR]]
  // CHECK: %[[GE:.+]] = spv.FOrdGreaterThanEqual %[[SUB]], %[[HALF]]
  // CHECK: %[[SEL:.+]] = spv.Select %[[GE]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[ADD:.+]] = spv.FAdd %[[FLOOR]], %[[SEL]]
  // CHECK: %[[BITCAST:.+]] = spv.Bitcast %[[ADD]]
  %0 = math.round %x : f32
  return %0: f32
}

// CHECK-LABEL: @round_vector
func.func @round_vector(%x: vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[ZERO:.+]] = spv.Constant dense<0.000000e+00>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00>
  // CHECK: %[[HALF:.+]] = spv.Constant dense<5.000000e-01>
  // CHECK: %[[ABS:.+]] = spv.GL.FAbs %arg0
  // CHECK: %[[FLOOR:.+]] = spv.GL.Floor %[[ABS]]
  // CHECK: %[[SUB:.+]] = spv.FSub %[[ABS]], %[[FLOOR]]
  // CHECK: %[[GE:.+]] = spv.FOrdGreaterThanEqual %[[SUB]], %[[HALF]]
  // CHECK: %[[SEL:.+]] = spv.Select %[[GE]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[ADD:.+]] = spv.FAdd %[[FLOOR]], %[[SEL]]
  // CHECK: %[[BITCAST:.+]] = spv.Bitcast %[[ADD]]
  %0 = math.round %x : vector<4xf32>
  return %0: vector<4xf32>
}

} // end module

// -----

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, Int64, Int16], []>, #spv.resource_limits<>>
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
