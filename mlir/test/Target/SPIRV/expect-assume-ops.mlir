// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires
  #spirv.vce<v1.0, [Shader, Linkage, ExpectAssumeKHR], [SPV_KHR_expect_assume]> {

  // CHECK-LABEL: @assume_true
  spirv.func @assume_true(%arg : i1) "None" {
    // CHECK: spirv.KHR.AssumeTrue %{{.*}}
    spirv.KHR.AssumeTrue %arg
    spirv.Return
  }

  // CHECK-LABEL: @expect_scalar_int
  spirv.func @expect_scalar_int(%val : i32, %expected : i32) -> i32 "None" {
    // CHECK: {{%.+}} = spirv.KHR.Expect %{{.*}}, %{{.*}} : i32
    %0 = spirv.KHR.Expect %val, %expected : i32
    spirv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @expect_scalar_bool
  spirv.func @expect_scalar_bool(%val : i1, %expected : i1) -> i1 "None" {
    // CHECK: {{%.+}} = spirv.KHR.Expect %{{.*}}, %{{.*}} : i1
    %0 = spirv.KHR.Expect %val, %expected : i1
    spirv.ReturnValue %0 : i1
  }

  // CHECK-LABEL: @expect_vector_int
  spirv.func @expect_vector_int(%val : vector<4xi32>, %expected : vector<4xi32>) -> vector<4xi32> "None" {
    // CHECK: {{%.+}} = spirv.KHR.Expect %{{.*}}, %{{.*}} : vector<4xi32>
    %0 = spirv.KHR.Expect %val, %expected : vector<4xi32>
    spirv.ReturnValue %0 : vector<4xi32>
  }

  // CHECK-LABEL: @expect_vector_bool
  spirv.func @expect_vector_bool(%val : vector<4xi1>, %expected : vector<4xi1>) -> vector<4xi1> "None" {
    // CHECK: {{%.+}} = spirv.KHR.Expect %{{.*}}, %{{.*}} : vector<4xi1>
    %0 = spirv.KHR.Expect %val, %expected : vector<4xi1>
    spirv.ReturnValue %0 : vector<4xi1>
  }
}
