// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip --split-input-file %s | FileCheck %s
// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires
    #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_linkonce_odr]> {
  // CHECK: spirv.func @linkonce_odr_fn() "None" attributes
  // CHECK-SAME: linkage_attributes = #spirv.linkage_attributes<linkage_name = "linkonce_odr_fn", linkage_type = <LinkOnceODR>>
  spirv.func @linkonce_odr_fn() "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<
      linkage_name = "linkonce_odr_fn",
      linkage_type = <LinkOnceODR>>
  } {
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires
    #spirv.vce<v1.0, [Shader, WeakLinkageAMD], [SPV_AMD_weak_linkage]> {
  // CHECK: spirv.func @weak_fn() "None" attributes
  // CHECK-SAME: linkage_attributes = #spirv.linkage_attributes<linkage_name = "weak_fn", linkage_type = <Weak>>
  spirv.func @weak_fn() "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<
      linkage_name = "weak_fn",
      linkage_type = <Weak>>
  } {
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires
    #spirv.vce<v1.0, [Shader, WeakLinkageAMD], [SPV_AMD_weak_linkage]> {
  // CHECK: spirv.GlobalVariable @weak_var
  // CHECK-SAME: linkage_attributes = #spirv.linkage_attributes<linkage_name = "weak_var", linkage_type = <Weak>>
  spirv.GlobalVariable @weak_var {
    linkage_attributes = #spirv.linkage_attributes<
      linkage_name = "weak_var",
      linkage_type = <Weak>>
  } : !spirv.ptr<i32, Private>
}
