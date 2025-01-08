// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// CHECK:      spirv.GlobalVariable @var0 bind(1, 0) : !spirv.ptr<f32, Input>
// CHECK-NEXT: spirv.GlobalVariable @var1 bind(0, 1) : !spirv.ptr<f32, Output>
// CHECK-NEXT: spirv.GlobalVariable @var2 built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
// CHECK-NEXT: spirv.GlobalVariable @var3 built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @var0 bind(1, 0) : !spirv.ptr<f32, Input>
  spirv.GlobalVariable @var1 bind(0, 1) : !spirv.ptr<f32, Output>
  spirv.GlobalVariable @var2 {built_in = "GlobalInvocationId"} : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @var3 built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK:         spirv.GlobalVariable @var1 : !spirv.ptr<f32, Input>
  // CHECK-NEXT:    spirv.GlobalVariable @var2 initializer(@var1) bind(1, 0) : !spirv.ptr<f32, Input>
  spirv.GlobalVariable @var1 : !spirv.ptr<f32, Input>
  spirv.GlobalVariable @var2 initializer(@var1) bind(1, 0) : !spirv.ptr<f32, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK:         spirv.SpecConstant @sc = 1 : i8
  // CHECK-NEXT:    spirv.GlobalVariable @var initializer(@sc) : !spirv.ptr<i8, Uniform>
  spirv.SpecConstant @sc = 1 : i8

  spirv.GlobalVariable @var initializer(@sc) : !spirv.ptr<i8, Uniform>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK:         spirv.SpecConstantComposite @scc (@sc0, @sc1, @sc2) : !spirv.array<3 x i8>
  // CHECK-NEXT:    spirv.GlobalVariable @var initializer(@scc) : !spirv.ptr<!spirv.array<3 x i8>, Uniform>
  spirv.SpecConstant @sc0 = 1 : i8
  spirv.SpecConstant @sc1 = 2 : i8
  spirv.SpecConstant @sc2 = 3 : i8

  spirv.SpecConstantComposite @scc (@sc0, @sc1, @sc2) : !spirv.array<3 x i8>

  spirv.GlobalVariable @var initializer(@scc) : !spirv.ptr<!spirv.array<3 x i8>, Uniform>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @globalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.func @foo() "None" {
    // CHECK: %[[ADDR:.*]] = spirv.mlir.addressof @globalInvocationID : !spirv.ptr<vector<3xi32>, Input>
    %0 = spirv.mlir.addressof @globalInvocationID : !spirv.ptr<vector<3xi32>, Input>
    %1 = spirv.Constant 0: i32
    // CHECK: spirv.AccessChain %[[ADDR]]
    %2 = spirv.AccessChain %0[%1] : !spirv.ptr<vector<3xi32>, Input>, i32 -> !spirv.ptr<i32, Input>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: linkage_attributes = #spirv.linkage_attributes<linkage_name = "outSideGlobalVar1", linkage_type = <Import>>
  spirv.GlobalVariable @var1 {
    linkage_attributes=#spirv.linkage_attributes<
      linkage_name="outSideGlobalVar1", 
      linkage_type=<Import>
    >
  } : !spirv.ptr<f32, Private>
}
