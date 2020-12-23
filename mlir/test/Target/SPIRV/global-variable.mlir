// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// CHECK:      spv.globalVariable @var0 bind(1, 0) : !spv.ptr<f32, Input>
// CHECK-NEXT: spv.globalVariable @var1 bind(0, 1) : !spv.ptr<f32, Output>
// CHECK-NEXT: spv.globalVariable @var2 built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
// CHECK-NEXT: spv.globalVariable @var3 built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.globalVariable @var0 bind(1, 0) : !spv.ptr<f32, Input>
  spv.globalVariable @var1 bind(0, 1) : !spv.ptr<f32, Output>
  spv.globalVariable @var2 {built_in = "GlobalInvocationId"} : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @var3 built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK:         spv.globalVariable @var1 : !spv.ptr<f32, Input>
  // CHECK-NEXT:    spv.globalVariable @var2 initializer(@var1) bind(1, 0) : !spv.ptr<f32, Input>
  spv.globalVariable @var1 : !spv.ptr<f32, Input>
  spv.globalVariable @var2 initializer(@var1) bind(1, 0) : !spv.ptr<f32, Input>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  spv.func @foo() "None" {
    // CHECK: %[[ADDR:.*]] = spv.mlir.addressof @globalInvocationID : !spv.ptr<vector<3xi32>, Input>
    %0 = spv.mlir.addressof @globalInvocationID : !spv.ptr<vector<3xi32>, Input>
    %1 = spv.constant 0: i32
    // CHECK: spv.AccessChain %[[ADDR]]
    %2 = spv.AccessChain %0[%1] : !spv.ptr<vector<3xi32>, Input>, i32
    spv.Return
  }
}
