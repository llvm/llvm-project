// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

// CHECK:      spirv.GlobalVariable @var0 bind(1, 0) : !spirv.ptr<f32, Input>
// CHECK-NEXT: spirv.GlobalVariable @var1 bind(0, 1) : !spirv.ptr<f32, Output>
// CHECK-NEXT: spirv.GlobalVariable @var2 built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
// CHECK-NEXT: spirv.GlobalVariable @var3 built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  spirv.GlobalVariable @var0 bind(1, 0) : !spirv.ptr<f32, Input>
  spirv.GlobalVariable @var1 bind(0, 1) : !spirv.ptr<f32, Output>
  spirv.GlobalVariable @var2 {built_in = "GlobalInvocationId"} : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @var3 built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
}

// -----

// Alignment and MaxByteOffset are also preserved on module-scope variables.
spirv.module Physical64 OpenCL requires #spirv.vce<v1.1, [Addresses, Kernel, Linkage], []> {
  // CHECK: spirv.GlobalVariable @aligned {alignment = 16 : i32} : !spirv.ptr<f32, CrossWorkgroup>
  // CHECK: spirv.GlobalVariable @high_bit {alignment = -2147483648 : i32} : !spirv.ptr<f32, CrossWorkgroup>
  // CHECK: spirv.GlobalVariable @bounded {max_byte_offset = 64 : i32} : !spirv.ptr<f32, CrossWorkgroup>
  spirv.GlobalVariable @aligned {alignment = 16 : i32} : !spirv.ptr<f32, CrossWorkgroup>
  spirv.GlobalVariable @high_bit {alignment = 2147483648 : i32} : !spirv.ptr<f32, CrossWorkgroup>
  spirv.GlobalVariable @bounded {max_byte_offset = 64 : i32} : !spirv.ptr<f32, CrossWorkgroup>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, Int8], []> {
  // CHECK:         spirv.SpecConstant @sc = 1 : i8
  // CHECK-NEXT:    spirv.GlobalVariable @var initializer(@sc) : !spirv.ptr<i8, Uniform>
  spirv.SpecConstant @sc = 1 : i8

  spirv.GlobalVariable @var initializer(@sc) : !spirv.ptr<i8, Uniform>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, Int8], []> {
  // CHECK:         spirv.SpecConstantComposite @scc (@sc0, @sc1, @sc2) : !spirv.array<3 x i8>
  // CHECK-NEXT:    spirv.GlobalVariable @var initializer(@scc) : !spirv.ptr<!spirv.array<3 x i8>, Uniform>
  spirv.SpecConstant @sc0 = 1 : i8
  spirv.SpecConstant @sc1 = 2 : i8
  spirv.SpecConstant @sc2 = 3 : i8

  spirv.SpecConstantComposite @scc (@sc0, @sc1, @sc2) : !spirv.array<3 x i8>

  spirv.GlobalVariable @var initializer(@scc) : !spirv.ptr<!spirv.array<3 x i8>, Uniform>
}

// -----

// Normal constants as initializers.

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK:         spirv.GlobalVariable @int_var initializer(42 : i32) : !spirv.ptr<i32, Private>
  // CHECK-NEXT:    spirv.GlobalVariable @float_var initializer(1.500000e+00 : f32) : !spirv.ptr<f32, Private>
  // CHECK-NEXT:    spirv.GlobalVariable @bool_var initializer(true) : !spirv.ptr<i1, Private>
  spirv.GlobalVariable @int_var initializer(42 : i32) : !spirv.ptr<i32, Private>
  spirv.GlobalVariable @float_var initializer(1.5 : f32) : !spirv.ptr<f32, Private>
  spirv.GlobalVariable @bool_var initializer(true) : !spirv.ptr<i1, Private>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK:         spirv.GlobalVariable @int_vec initializer(dense<[1, 2, 3]> : vector<3xi32>) : !spirv.ptr<vector<3xi32>, Private>
  // CHECK-NEXT:    spirv.GlobalVariable @float_vec initializer(dense<[1.000000e+00, 2.000000e+00]> : vector<2xf32>) : !spirv.ptr<vector<2xf32>, Private>
  spirv.GlobalVariable @int_vec initializer(dense<[1, 2, 3]> : vector<3xi32>) : !spirv.ptr<vector<3xi32>, Private>
  spirv.GlobalVariable @float_vec initializer(dense<[1.0, 2.0]> : vector<2xf32>) : !spirv.ptr<vector<2xf32>, Private>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK:         spirv.GlobalVariable @array_var initializer([1 : i32, 2 : i32, 3 : i32, 4 : i32]) : !spirv.ptr<!spirv.array<4 x i32>, Private>
  // CHECK-NEXT:    spirv.GlobalVariable @dense_array_var initializer([5 : i32, 6 : i32, 7 : i32]) : !spirv.ptr<!spirv.array<3 x i32>, Private>
  // CHECK-NEXT:    spirv.GlobalVariable @struct_var initializer([1 : i32, 2.000000e+00 : f32]) : !spirv.ptr<!spirv.struct<(i32, f32)>, Private>
  spirv.GlobalVariable @array_var initializer([1 : i32, 2 : i32, 3 : i32, 4 : i32]) : !spirv.ptr<!spirv.array<4 x i32>, Private>
  spirv.GlobalVariable @dense_array_var initializer(dense<[5, 6, 7]> : tensor<3xi32>) : !spirv.ptr<!spirv.array<3 x i32>, Private>
  spirv.GlobalVariable @struct_var initializer([1 : i32, 2.0 : f32]) : !spirv.ptr<!spirv.struct<(i32, f32)>, Private>
}

// -----

// Zero initializers.

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK:         spirv.GlobalVariable @zero_int initializer(0 : i32) : !spirv.ptr<i32, Private>
  // CHECK-NEXT:    spirv.GlobalVariable @zero_vec initializer(dense<0> : vector<4xi32>) : !spirv.ptr<vector<4xi32>, Private>
  // CHECK-NEXT:    spirv.GlobalVariable @zero_array initializer([0 : i32, 0 : i32]) : !spirv.ptr<!spirv.array<2 x i32>, Private>
  spirv.GlobalVariable @zero_int initializer(0 : i32) : !spirv.ptr<i32, Private>
  spirv.GlobalVariable @zero_vec initializer(dense<0> : vector<4xi32>) : !spirv.ptr<vector<4xi32>, Private>
  spirv.GlobalVariable @zero_array initializer([0 : i32, 0 : i32]) : !spirv.ptr<!spirv.array<2 x i32>, Private>
}

// -----

// A constant used both as an initializer and in a function.

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK:         spirv.GlobalVariable @var initializer(7 : i32) : !spirv.ptr<i32, Private>
  spirv.GlobalVariable @var initializer(7 : i32) : !spirv.ptr<i32, Private>
  // CHECK:         spirv.func @use_constant
  spirv.func @use_constant() -> i32 "None" {
    // CHECK:         spirv.Constant 7 : i32
    %0 = spirv.Constant 7 : i32
    spirv.ReturnValue %0 : i32
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
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

// -----

// Equal attributes must not share constant IDs across different SPIR-V types.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: spirv.GlobalVariable @array initializer([1 : i32, 2 : i32]) : !spirv.ptr<!spirv.array<2 x i32>, Private>
  spirv.GlobalVariable @array initializer([1 : i32, 2 : i32]) : !spirv.ptr<!spirv.array<2 x i32>, Private>
  // CHECK: spirv.GlobalVariable @struct initializer([1 : i32, 2 : i32]) : !spirv.ptr<!spirv.struct<(i32, i32)>, Private>
  spirv.GlobalVariable @struct initializer([1 : i32, 2 : i32]) : !spirv.ptr<!spirv.struct<(i32, i32)>, Private>
  // CHECK: spirv.GlobalVariable @vector initializer(dense<[3, 4]> : vector<2xi32>) : !spirv.ptr<vector<2xi32>, Private>
  spirv.GlobalVariable @vector initializer(dense<[3, 4]> : vector<2xi32>) : !spirv.ptr<vector<2xi32>, Private>
  // CHECK: spirv.GlobalVariable @dense_array initializer([3 : i32, 4 : i32]) : !spirv.ptr<!spirv.array<2 x i32>, Private>
  spirv.GlobalVariable @dense_array initializer(dense<[3, 4]> : vector<2xi32>) : !spirv.ptr<!spirv.array<2 x i32>, Private>
}
