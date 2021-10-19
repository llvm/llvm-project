// RUN: mlir-opt -split-input-file -convert-math-to-spirv -verify-diagnostics %s -o - | FileCheck %s

// CHECK-LABEL: @float32_unary_scalar
func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spv.GLSL.Cos %{{.*}}: f32
  %0 = math.cos %arg0 : f32
  // CHECK: spv.GLSL.Exp %{{.*}}: f32
  %1 = math.exp %arg0 : f32
  // CHECK: spv.GLSL.Log %{{.*}}: f32
  %2 = math.log %arg0 : f32
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f32
  // CHECK: %[[ADDONE:.+]] = spv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spv.GLSL.Log %[[ADDONE]]
  %3 = math.log1p %arg0 : f32
  // CHECK: spv.GLSL.InverseSqrt %{{.*}}: f32
  %4 = math.rsqrt %arg0 : f32
  // CHECK: spv.GLSL.Sqrt %{{.*}}: f32
  %5 = math.sqrt %arg0 : f32
  // CHECK: spv.GLSL.Tanh %{{.*}}: f32
  %6 = math.tanh %arg0 : f32
  // CHECK: spv.GLSL.Sin %{{.*}}: f32
  %7 = math.sin %arg0 : f32
  return
}

// CHECK-LABEL: @float32_unary_vector
func @float32_unary_vector(%arg0: vector<3xf32>) {
  // CHECK: spv.GLSL.Cos %{{.*}}: vector<3xf32>
  %0 = math.cos %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Exp %{{.*}}: vector<3xf32>
  %1 = math.exp %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Log %{{.*}}: vector<3xf32>
  %2 = math.log %arg0 : vector<3xf32>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: %[[ADDONE:.+]] = spv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spv.GLSL.Log %[[ADDONE]]
  %3 = math.log1p %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.InverseSqrt %{{.*}}: vector<3xf32>
  %4 = math.rsqrt %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Sqrt %{{.*}}: vector<3xf32>
  %5 = math.sqrt %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Tanh %{{.*}}: vector<3xf32>
  %6 = math.tanh %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Sin %{{.*}}: vector<3xf32>
  %7 = math.sin %arg0 : vector<3xf32>
  return
}

// CHECK-LABEL: @float32_binary_scalar
func @float32_binary_scalar(%lhs: f32, %rhs: f32) {
  // CHECK: spv.GLSL.Pow %{{.*}}: f32
  %0 = math.powf %lhs, %rhs : f32
  return
}

// CHECK-LABEL: @float32_binary_vector
func @float32_binary_vector(%lhs: vector<4xf32>, %rhs: vector<4xf32>) {
  // CHECK: spv.GLSL.Pow %{{.*}}: vector<4xf32>
  %0 = math.powf %lhs, %rhs : vector<4xf32>
  return
}
