// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Linkage, AbortKHR], [SPV_KHR_abort]> {
  // CHECK-LABEL: spirv.func @abort_i32
  spirv.func @abort_i32(%cond: i1) "None" {
    spirv.BranchConditional %cond, ^ok, ^abort
  ^ok:
    spirv.Return
  ^abort:
    // CHECK: %[[MSG:.+]] = spirv.Constant -1 : i32
    // CHECK: spirv.KHR.Abort %[[MSG]] : i32
    %msg = spirv.Constant -1 : i32
    spirv.KHR.Abort %msg : i32
  }

  // CHECK-LABEL: spirv.func @abort_struct
  spirv.func @abort_struct(%a: i32, %b: vector<2xf32>) "None" {
    // CHECK: %[[MSG:.+]] = spirv.CompositeConstruct
    // CHECK: spirv.KHR.Abort %[[MSG]] : !spirv.struct<(i32, vector<2xf32>)>
    %msg = spirv.CompositeConstruct %a, %b : (i32, vector<2xf32>) -> !spirv.struct<(i32, vector<2xf32>)>
    spirv.KHR.Abort %msg : !spirv.struct<(i32, vector<2xf32>)>
  }

  // CHECK-LABEL: spirv.func @abort_ptr
  spirv.func @abort_ptr(%p: !spirv.ptr<i32, CrossWorkgroup>) "None" {
    // CHECK: spirv.KHR.Abort %{{.*}} : !spirv.ptr<i32, CrossWorkgroup>
    spirv.KHR.Abort %p : !spirv.ptr<i32, CrossWorkgroup>
  }
}
