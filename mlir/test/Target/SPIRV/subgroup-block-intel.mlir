// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Physical64 GLSL450 requires #spirv.vce<v1.3, [Addresses, Shader, Linkage, SubgroupBufferBlockIOINTEL],
                                                          [SPV_KHR_storage_buffer_storage_class, SPV_INTEL_subgroups]> {
  // CHECK-LABEL: @subgroup_block_read_intel
  spirv.func @subgroup_block_read_intel(%ptr : !spirv.ptr<i32, StorageBuffer>) -> i32 "None" {
    // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : !spirv.ptr<i32, StorageBuffer> -> i32
    %0 = spirv.INTEL.SubgroupBlockRead %ptr : !spirv.ptr<i32, StorageBuffer> -> i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @subgroup_block_read_intel_vector
  spirv.func @subgroup_block_read_intel_vector(%ptr : !spirv.ptr<i32, StorageBuffer>) -> vector<3xi32> "None" {
    // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : !spirv.ptr<i32, StorageBuffer> -> vector<3xi32>
    %0 = spirv.INTEL.SubgroupBlockRead %ptr : !spirv.ptr<i32, StorageBuffer> -> vector<3xi32>
    spirv.ReturnValue %0: vector<3xi32>
  }
  // CHECK-LABEL: @subgroup_block_write_intel
  spirv.func @subgroup_block_write_intel(%ptr : !spirv.ptr<i32, StorageBuffer>, %value: i32) -> () "None" {
    // CHECK: spirv.INTEL.SubgroupBlockWrite %{{.*}}, %{{.*}} : i32
    spirv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : i32
    spirv.Return
  }
  // CHECK-LABEL: @subgroup_block_write_intel_vector
  spirv.func @subgroup_block_write_intel_vector(%ptr : !spirv.ptr<i32, StorageBuffer>, %value: vector<3xi32>) -> () "None" {
    // CHECK: spirv.INTEL.SubgroupBlockWrite %{{.*}}, %{{.*}} : vector<3xi32>
    spirv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : vector<3xi32>
    spirv.Return
  }
}
