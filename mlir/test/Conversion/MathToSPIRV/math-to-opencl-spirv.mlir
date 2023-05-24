// RUN: mlir-opt -split-input-file -convert-math-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes { spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>> } {

// CHECK-LABEL: @float32_unary_scalar
func.func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spirv.CL.cos %{{.*}}: f32
  %0 = math.cos %arg0 : f32
  // CHECK: spirv.CL.exp %{{.*}}: f32
  %1 = math.exp %arg0 : f32
  // CHECK: %[[EXP:.+]] = spirv.CL.exp %arg0
  // CHECK: %[[ONE:.+]] = spirv.Constant 1.000000e+00 : f32
  // CHECK: spirv.FSub %[[EXP]], %[[ONE]]
  %2 = math.expm1 %arg0 : f32
  // CHECK: spirv.CL.log %{{.*}}: f32
  %3 = math.log %arg0 : f32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1.000000e+00 : f32
  // CHECK: %[[ADDONE:.+]] = spirv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spirv.CL.log %[[ADDONE]]
  %4 = math.log1p %arg0 : f32
  // CHECK: spirv.CL.rint %{{.*}}: f32
  %5 = math.roundeven %arg0 : f32
  // CHECK: spirv.CL.rsqrt %{{.*}}: f32
  %6 = math.rsqrt %arg0 : f32
  // CHECK: spirv.CL.sqrt %{{.*}}: f32
  %7 = math.sqrt %arg0 : f32
  // CHECK: spirv.CL.tanh %{{.*}}: f32
  %8 = math.tanh %arg0 : f32
  // CHECK: spirv.CL.sin %{{.*}}: f32
  %9 = math.sin %arg0 : f32
  // CHECK: spirv.CL.fabs %{{.*}}: f32
  %10 = math.absf %arg0 : f32
  // CHECK: spirv.CL.ceil %{{.*}}: f32
  %11 = math.ceil %arg0 : f32
  // CHECK: spirv.CL.floor %{{.*}}: f32
  %12 = math.floor %arg0 : f32
  // CHECK: spirv.CL.erf %{{.*}}: f32
  %13 = math.erf %arg0 : f32
  // CHECK: spirv.CL.round %{{.*}}: f32
  %14 = math.round %arg0 : f32
  return
}

// CHECK-LABEL: @float32_unary_vector
func.func @float32_unary_vector(%arg0: vector<3xf32>) {
  // CHECK: spirv.CL.cos %{{.*}}: vector<3xf32>
  %0 = math.cos %arg0 : vector<3xf32>
  // CHECK: spirv.CL.exp %{{.*}}: vector<3xf32>
  %1 = math.exp %arg0 : vector<3xf32>
  // CHECK: %[[EXP:.+]] = spirv.CL.exp %arg0
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: spirv.FSub %[[EXP]], %[[ONE]]
  %2 = math.expm1 %arg0 : vector<3xf32>
  // CHECK: spirv.CL.log %{{.*}}: vector<3xf32>
  %3 = math.log %arg0 : vector<3xf32>
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: %[[ADDONE:.+]] = spirv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spirv.CL.log %[[ADDONE]]
  %4 = math.log1p %arg0 : vector<3xf32>
  // CHECK: spirv.CL.rint %{{.*}}: vector<3xf32>
  %5 = math.roundeven %arg0 : vector<3xf32>
  // CHECK: spirv.CL.rsqrt %{{.*}}: vector<3xf32>
  %6 = math.rsqrt %arg0 : vector<3xf32>
  // CHECK: spirv.CL.sqrt %{{.*}}: vector<3xf32>
  %7 = math.sqrt %arg0 : vector<3xf32>
  // CHECK: spirv.CL.tanh %{{.*}}: vector<3xf32>
  %8 = math.tanh %arg0 : vector<3xf32>
  // CHECK: spirv.CL.sin %{{.*}}: vector<3xf32>
  %9 = math.sin %arg0 : vector<3xf32>
  return
}

// CHECK-LABEL: @float32_binary_scalar
func.func @float32_binary_scalar(%lhs: f32, %rhs: f32) {
  // CHECK: spirv.CL.pow %{{.*}}: f32
  %0 = math.powf %lhs, %rhs : f32
  return
}

// CHECK-LABEL: @float32_binary_vector
func.func @float32_binary_vector(%lhs: vector<4xf32>, %rhs: vector<4xf32>) {
  // CHECK: spirv.CL.pow %{{.*}}: vector<4xf32>
  %0 = math.powf %lhs, %rhs : vector<4xf32>
  return
}

// CHECK-LABEL: @float32_ternary_scalar
func.func @float32_ternary_scalar(%a: f32, %b: f32, %c: f32) {
  // CHECK: spirv.CL.fma %{{.*}}: f32
  %0 = math.fma %a, %b, %c : f32
  return
}

// CHECK-LABEL: @float32_ternary_vector
func.func @float32_ternary_vector(%a: vector<4xf32>, %b: vector<4xf32>,
                            %c: vector<4xf32>) {
  // CHECK: spirv.CL.fma %{{.*}}: vector<4xf32>
  %0 = math.fma %a, %b, %c : vector<4xf32>
  return
}

// CHECK-LABEL: @int_unary
func.func @int_unary(%arg0: i32) {
  // CHECK: spirv.CL.s_abs %{{.*}}
  %0 = math.absi %arg0 : i32
  return
}

} // end module

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {

// 2-D vectors are not supported.

// CHECK-LABEL: @vector_2d
func.func @vector_2d(%arg0: vector<2x2xf32>) {
  // CHECK-NEXT: math.cos {{.+}} : vector<2x2xf32>
  %0 = math.cos %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.exp {{.+}} : vector<2x2xf32>
  %1 = math.exp %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.absf {{.+}} : vector<2x2xf32>
  %2 = math.absf %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.ceil {{.+}} : vector<2x2xf32>
  %3 = math.ceil %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.floor {{.+}} : vector<2x2xf32>
  %4 = math.floor %arg0 : vector<2x2xf32>
  // CHECK-NEXT: math.powf {{.+}}, {{%.+}} : vector<2x2xf32>
  %5 = math.powf %arg0, %arg0 : vector<2x2xf32>
  // CHECK-NEXT: return
  return
}

// Tensors are not supported.

// CHECK-LABEL: @tensor_1d
func.func @tensor_1d(%arg0: tensor<2xf32>) {
  // CHECK-NEXT: math.cos {{.+}} : tensor<2xf32>
  %0 = math.cos %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.exp {{.+}} : tensor<2xf32>
  %1 = math.exp %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.absf {{.+}} : tensor<2xf32>
  %2 = math.absf %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.ceil {{.+}} : tensor<2xf32>
  %3 = math.ceil %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.floor {{.+}} : tensor<2xf32>
  %4 = math.floor %arg0 : tensor<2xf32>
  // CHECK-NEXT: math.powf {{.+}}, {{%.+}} : tensor<2xf32>
  %5 = math.powf %arg0, %arg0 : tensor<2xf32>
  // CHECK-NEXT: return
  return
}

} // end module
