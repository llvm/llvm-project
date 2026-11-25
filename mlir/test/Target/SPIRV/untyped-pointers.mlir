// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Linkage, UntypedPointersKHR], [SPV_KHR_untyped_pointers]> {
  // CHECK: spirv.SpecConstant @sc = 1.000000e+00 : f32
  spirv.SpecConstant @sc = 1.0 : f32
  // CHECK: spirv.GlobalVariable @init_var data_type(f32) initializer(@sc) : !spirv.untyped_ptr<CrossWorkgroup>
  spirv.GlobalVariable @init_var data_type(f32) initializer(@sc) : !spirv.untyped_ptr<CrossWorkgroup>
  // CHECK: spirv.GlobalVariable @wg_var data_type(!spirv.array<4 x f32>) : !spirv.untyped_ptr<Workgroup>
  spirv.GlobalVariable @wg_var data_type(!spirv.array<4 x f32>) : !spirv.untyped_ptr<Workgroup>
  // CHECK: spirv.GlobalVariable @cwg_var : !spirv.untyped_ptr<CrossWorkgroup>
  spirv.GlobalVariable @cwg_var : !spirv.untyped_ptr<CrossWorkgroup>

  // CHECK-LABEL: spirv.func @untyped_ptr_arg
  // CHECK-SAME: !spirv.untyped_ptr<CrossWorkgroup>
  // CHECK-SAME: !spirv.untyped_ptr<Workgroup>
  spirv.func @untyped_ptr_arg(%arg0: !spirv.untyped_ptr<CrossWorkgroup>, %arg1: !spirv.untyped_ptr<Workgroup>) "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<linkage_name = "untyped_ptr_arg", linkage_type = <Export>>
  } {
    spirv.Return
  }

  // CHECK-LABEL: spirv.func @untyped_load_store
  spirv.func @untyped_load_store(%arg0: !spirv.untyped_ptr<CrossWorkgroup>, %arg1: !spirv.untyped_ptr<Workgroup>) "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<linkage_name = "untyped_load_store", linkage_type = <Export>>
  } {
    // CHECK: %[[V:.*]] = spirv.Load %{{.*}} ["Aligned", 4] : !spirv.untyped_ptr<CrossWorkgroup> -> f32
    %0 = spirv.Load %arg0 ["Aligned", 4] : !spirv.untyped_ptr<CrossWorkgroup> -> f32
    // CHECK: spirv.Store %{{.*}}, %[[V]] : !spirv.untyped_ptr<Workgroup>, f32
    spirv.Store %arg1, %0 : !spirv.untyped_ptr<Workgroup>, f32
    spirv.Return
  }

  // CHECK-LABEL: spirv.func @untyped_global
  spirv.func @untyped_global(%arg0: f32) "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<linkage_name = "untyped_global", linkage_type = <Export>>
  } {
    // CHECK: %[[WG:.*]] = spirv.mlir.addressof @wg_var : !spirv.untyped_ptr<Workgroup>
    // CHECK: spirv.Store %[[WG]], %{{.*}} ["Aligned", 4] : !spirv.untyped_ptr<Workgroup>, f32
    %0 = spirv.mlir.addressof @wg_var : !spirv.untyped_ptr<Workgroup>
    spirv.Store %0, %arg0 ["Aligned", 4] : !spirv.untyped_ptr<Workgroup>, f32
    // CHECK: %[[CWG:.*]] = spirv.mlir.addressof @cwg_var : !spirv.untyped_ptr<CrossWorkgroup>
    // CHECK: spirv.Load %[[CWG]] : !spirv.untyped_ptr<CrossWorkgroup> -> i32
    %1 = spirv.mlir.addressof @cwg_var : !spirv.untyped_ptr<CrossWorkgroup>
    %2 = spirv.Load %1 : !spirv.untyped_ptr<CrossWorkgroup> -> i32
    spirv.Return
  }
}
