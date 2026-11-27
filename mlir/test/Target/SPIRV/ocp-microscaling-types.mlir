// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0,
    [Shader, Linkage, Float4EXT, Float6EXT, Float8UnsignedE8M0EXT, BitcastExtractEXT],
    [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types]> {
  // CHECK: spirv.GlobalVariable @fp4 : !spirv.ptr<f4E2M1FN, Private>
  spirv.GlobalVariable @fp4 : !spirv.ptr<f4E2M1FN, Private>
  // CHECK: spirv.GlobalVariable @fp6e2m3 : !spirv.ptr<f6E2M3FN, Private>
  spirv.GlobalVariable @fp6e2m3 : !spirv.ptr<f6E2M3FN, Private>
  // CHECK: spirv.GlobalVariable @fp6e3m2 : !spirv.ptr<f6E3M2FN, Private>
  spirv.GlobalVariable @fp6e3m2 : !spirv.ptr<f6E3M2FN, Private>
  // CHECK: spirv.GlobalVariable @fp8e8m0 : !spirv.ptr<f8E8M0FNU, Workgroup>
  spirv.GlobalVariable @fp8e8m0 : !spirv.ptr<f8E8M0FNU, Workgroup>

  // CHECK-LABEL: @microscaling_types
  // CHECK-SAME: f4E2M1FN
  // CHECK-SAME: f6E2M3FN
  // CHECK-SAME: f6E3M2FN
  // CHECK-SAME: f8E8M0FNU
  spirv.func @microscaling_types(%f4: f4E2M1FN, %f6e2m3: f6E2M3FN,
                                %f6e3m2: f6E3M2FN, %e8m0: f8E8M0FNU)
      -> () "None" {
    spirv.Return
  }

  // CHECK-LABEL: @microscaling_constants
  // CHECK: spirv.Constant -1.500000e+00 : f4E2M1FN
  // CHECK: spirv.Constant -1.750000e+00 : f6E2M3FN
  // CHECK: spirv.Constant -1.250000e+00 : f6E3M2FN
  // CHECK: spirv.Constant 8.000000e+00 : f8E8M0FNU
  spirv.func @microscaling_constants() -> () "None" {
    %0 = spirv.Constant -1.5 : f4E2M1FN
    %1 = spirv.Constant -1.75 : f6E2M3FN
    %2 = spirv.Constant -1.25 : f6E3M2FN
    %3 = spirv.Constant 8.0 : f8E8M0FNU
    %4 = spirv.FConvert %0 : f4E2M1FN to f32
    %5 = spirv.FConvert %1 : f6E2M3FN to f32
    %6 = spirv.FConvert %2 : f6E3M2FN to f32
    %7 = spirv.FConvert %3 : f8E8M0FNU to f32
    spirv.Return
  }

  // Use distinct elements to exercise non-splat composite constants.
  // CHECK-LABEL: @microscaling_vector_constants
  // CHECK: spirv.Constant dense<[-1.500000e+00, 5.000000e-01]> : vector<2xf4E2M1FN>
  // CHECK: spirv.Constant dense<[-1.750000e+00, 5.000000e-01]> : vector<2xf6E2M3FN>
  // CHECK: spirv.Constant dense<[-1.250000e+00, 5.000000e-01]> : vector<2xf6E3M2FN>
  // CHECK: spirv.Constant dense<[8.000000e+00, 5.000000e-01]> : vector<2xf8E8M0FNU>
  spirv.func @microscaling_vector_constants() "None" {
    %0 = spirv.Constant dense<[-1.5, 0.5]> : vector<2xf4E2M1FN>
    %1 = spirv.Constant dense<[-1.75, 0.5]> : vector<2xf6E2M3FN>
    %2 = spirv.Constant dense<[-1.25, 0.5]> : vector<2xf6E3M2FN>
    %3 = spirv.Constant dense<[8.0, 0.5]> : vector<2xf8E8M0FNU>
    %4 = spirv.FConvert %0 : vector<2xf4E2M1FN> to vector<2xf32>
    %5 = spirv.FConvert %1 : vector<2xf6E2M3FN> to vector<2xf32>
    %6 = spirv.FConvert %2 : vector<2xf6E3M2FN> to vector<2xf32>
    %7 = spirv.FConvert %3 : vector<2xf8E8M0FNU> to vector<2xf32>
    spirv.Return
  }

  // CHECK-LABEL: @bitcast_extract
  // CHECK: spirv.BitcastExtractEXT {{%.*}}, {{%.*}} : i32, i32 to f4E2M1FN
  // CHECK: spirv.BitcastExtractEXT {{%.*}}, {{%.*}} : vector<2xi32>, i32 to vector<2xf6E3M2FN>
  // CHECK: spirv.BitcastExtractEXT {{%.*}}, {{%.*}} : i32, i32 to f6E2M3FN
  // CHECK: spirv.BitcastExtractEXT {{%.*}}, {{%.*}} : vector<2xi32>, i32 to vector<2xf8E8M0FNU>
  spirv.func @bitcast_extract(%base: i32, %vector: vector<2xi32>, %offset: i32)
      -> () "None" {
    %0 = spirv.BitcastExtractEXT %base, %offset : i32, i32 to f4E2M1FN
    %1 = spirv.BitcastExtractEXT %vector, %offset : vector<2xi32>, i32 to vector<2xf6E3M2FN>
    %2 = spirv.BitcastExtractEXT %base, %offset : i32, i32 to f6E2M3FN
    %3 = spirv.BitcastExtractEXT %vector, %offset : vector<2xi32>, i32 to vector<2xf8E8M0FNU>
    spirv.Return
  }
}
