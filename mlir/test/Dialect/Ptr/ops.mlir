// RUN: mlir-opt %s --verify-roundtrip | FileCheck %s

/// Check op assembly.
// CHECK-LABEL: @ptr_add_type_offset
func.func @ptr_add_type_offset(%ptr: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  // CHECK: ptr.type_offset f32 : index
  // CHECK-NEXT: ptr.ptr_add %{{.*}}, %{{.*}} : <#ptr.generic_space>, index
  // CHECK-NEXT: ptr.ptr_add %{{.*}}, %{{.*}} : <#ptr.generic_space>, index
  // CHECK-NEXT: ptr.ptr_add nusw %{{.*}}, %{{.*}} : <#ptr.generic_space>, index
  // CHECK-NEXT: ptr.ptr_add nuw %{{.*}}, %{{.*}} : <#ptr.generic_space>, index
  // CHECK-NEXT: ptr.ptr_add inbounds %{{.*}}, %{{.*}} : <#ptr.generic_space>, index
  %off = ptr.type_offset f32 : index
  %res = ptr.ptr_add %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res0 = ptr.ptr_add none %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res1 = ptr.ptr_add nusw %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res2 = ptr.ptr_add nuw %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res3 = ptr.ptr_add inbounds %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  return %res : !ptr.ptr<#ptr.generic_space>
}
