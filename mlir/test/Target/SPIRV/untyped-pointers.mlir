// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Linkage, UntypedPointersKHR], [SPV_KHR_untyped_pointers]> {
  // CHECK-LABEL: spirv.func @untyped_ptr_arg
  // CHECK-SAME: !spirv.untyped_ptr<CrossWorkgroup>
  // CHECK-SAME: !spirv.untyped_ptr<Workgroup>
  spirv.func @untyped_ptr_arg(%arg0: !spirv.untyped_ptr<CrossWorkgroup>, %arg1: !spirv.untyped_ptr<Workgroup>) "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<linkage_name = "untyped_ptr_arg", linkage_type = <Export>>
  } {
    spirv.Return
  }
}
