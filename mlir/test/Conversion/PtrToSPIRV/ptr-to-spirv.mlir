// RUN: mlir-opt %s -split-input-file -convert-ptr-to-spirv | FileCheck %s

!ptr = !ptr.ptr<#ptr.generic_space>

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.5, [Shader, Int64, PhysicalStorageBufferAddresses],
               [SPV_KHR_physical_storage_buffer]>,
    #spirv.resource_limits<>>
} {

// CHECK-LABEL: func.func @ptr_add_load_store
// CHECK-SAME:    %[[PTR:.*]]: !ptr.ptr<#ptr.generic_space>
// CHECK-SAME:    %[[OFFSET:.*]]: index
// CHECK-SAME:    %[[VALUE:.*]]: f32
func.func @ptr_add_load_store(%ptr: !ptr, %offset: index, %value: f32) -> f32 {
  // CHECK-DAG: %[[ADDR:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !ptr.ptr<#ptr.generic_space> to i64
  // CHECK-DAG: %[[SPIRV_OFFSET:.*]] = builtin.unrealized_conversion_cast %[[OFFSET]] : index to i32
  // CHECK: %[[OFFSET64:.*]] = spirv.UConvert %[[SPIRV_OFFSET]] : i32 to i64
  // CHECK: %[[ELEM_ADDR:.*]] = spirv.IAdd %[[ADDR]], %[[OFFSET64]] : i64
  %elem = ptr.ptr_add %ptr, %offset : !ptr, index

  // CHECK: %[[LOAD_PTR:.*]] = spirv.ConvertUToPtr %[[ELEM_ADDR]] : i64 to !spirv.ptr<f32, PhysicalStorageBuffer>
  // CHECK: %[[LOADED:.*]] = spirv.Load "PhysicalStorageBuffer" %[[LOAD_PTR]] ["Aligned", 4] : f32
  %loaded = ptr.load %elem alignment = 4 : !ptr -> f32

  // CHECK: %[[STORE_PTR:.*]] = spirv.ConvertUToPtr %[[ELEM_ADDR]] : i64 to !spirv.ptr<f32, PhysicalStorageBuffer>
  // CHECK: spirv.Store "PhysicalStorageBuffer" %[[STORE_PTR]], %[[VALUE]] ["Aligned", 4] : f32
  ptr.store %value, %elem alignment = 4 : f32, !ptr

  // CHECK: return %[[LOADED]] : f32
  return %loaded : f32
}

// CHECK-LABEL: func.func @type_offset
func.func @type_offset() -> index {
  // CHECK: %[[SIZE:.*]] = spirv.Constant 4 : i32
  %size = ptr.type_offset f32 : index
  // CHECK: builtin.unrealized_conversion_cast %[[SIZE]] : i32 to index
  return %size : index
}

} // module
