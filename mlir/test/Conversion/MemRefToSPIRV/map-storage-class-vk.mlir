// RUN: mlir-opt --allow-unregistered-dialect --map-memref-spirv-storage-class='client-api=vulkan' %s | FileCheck %s

// Vulkan Specific Mappings:
//   8 -> UniformConstant
//   9 -> Input
//   10 -> Output
//   11 -> PhysicalStorageBuffer
//   12 -> Image

/// Check that Vulkan specific memory space indices get converted into the correct
/// SPIR-V storage class. If mappings to OpenCL address spaces are added for these
/// indices then those test case should be moved into the common test file.

// CHECK-LABEL: func @test_vk_specific_memory_spaces
func.func @test_vk_specific_memory_spaces() {
  // CHECK: memref<4xi32, #spirv.storage_class<UniformConstant>>
  %1 = "dialect.memref_producer"() : () -> (memref<4xi32, 8>)
  // CHECK: memref<4xi32, #spirv.storage_class<Input>>
  %2 = "dialect.memref_producer"() : () -> (memref<4xi32, 9>)
  // CHECK: memref<4xi32, #spirv.storage_class<Output>>
  %3 = "dialect.memref_producer"() : () -> (memref<4xi32, 10>)
  // CHECK: memref<4xi32, #spirv.storage_class<PhysicalStorageBuffer>>
  %4 = "dialect.memref_producer"() : () -> (memref<4xi32, 11>)
  // CHECK: memref<4xi32, #spirv.storage_class<Image>>
  %5 = "dialect.memref_producer"() : () -> (memref<4xi32, 12>)
  return
}
