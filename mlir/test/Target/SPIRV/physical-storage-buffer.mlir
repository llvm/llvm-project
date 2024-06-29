// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s

// Test file showing how the Physical Storage Buffer extension works end-2-end.

!f32_binding = !spirv.struct<binding_f32_t, (!spirv.rtarray<f32, stride=4> [0])>
!f32_binding_ptr = !spirv.ptr<!f32_binding, PhysicalStorageBuffer>

!set_0 = !spirv.struct<set_0_t, (!f32_binding_ptr [0],
                                 !f32_binding_ptr [8],
                                 !f32_binding_ptr [16])>
!set_0_ptr = !spirv.ptr<!set_0, StorageBuffer>

!set_1 = !spirv.struct<set_1_t, (!f32_binding_ptr [0],
                                 !f32_binding_ptr [8])>
!set_1_ptr = !spirv.ptr<!set_1, StorageBuffer>

spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.5,
    [Shader, Int64, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {

  spirv.GlobalVariable @set_0 bind(3, 0) : !set_0_ptr
  spirv.GlobalVariable @set_1 bind(3, 1) : !set_1_ptr

  // CHECK-LABEL: spirv.func @main() "None"
  spirv.func @main() "None" {
    %idx0 = spirv.Constant 0 : i64
    %idx1 = spirv.Constant 1 : i64
    %idx2 = spirv.Constant 2 : i64
    %set_0_addr = spirv.mlir.addressof @set_0 : !set_0_ptr
    %s0_b2_ptr = spirv.AccessChain %set_0_addr[%idx2] : !set_0_ptr, i64
    %b2_ptr = spirv.Load "StorageBuffer" %s0_b2_ptr : !f32_binding_ptr
    %b2_data_ptr = spirv.AccessChain %b2_ptr[%idx0, %idx0] : !f32_binding_ptr, i64, i64

    // CHECK: spirv.Load "PhysicalStorageBuffer"
    %b2_data = spirv.Load "PhysicalStorageBuffer" %b2_data_ptr ["Aligned", 4] : f32

    %set_1_addr = spirv.mlir.addressof @set_1 : !set_1_ptr
    %s1_b1_ptr = spirv.AccessChain %set_1_addr[%idx1] : !set_1_ptr, i64
    %b1_ptr = spirv.Load "StorageBuffer" %s1_b1_ptr : !f32_binding_ptr
    %b1_data_ptr = spirv.AccessChain %b1_ptr[%idx0, %idx0] : !f32_binding_ptr, i64, i64

    // CHECK: spirv.Store "PhysicalStorageBuffer"
    spirv.Store "PhysicalStorageBuffer" %b1_data_ptr, %b2_data ["Aligned", 4] : f32

    spirv.Return
  }

  spirv.EntryPoint "GLCompute" @main, @set_0, @set_1
}
