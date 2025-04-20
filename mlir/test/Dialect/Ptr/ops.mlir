// RUN: mlir-opt %s --verify-roundtrip | FileCheck %s

/// Check op assembly.
func.func @ops0(%ptr: !ptr.ptr<#ptr.int_space>) -> !ptr.ptr<#ptr.int_space> {
  // CHECK-LABEL: @ops0
  // CHECK: ptr.type_offset f32 : index
  // CHECK-NEXT: ptr.ptradd %{{.*}}, %{{.*}} : <#ptr.int_space>, index
  %off = ptr.type_offset f32 : index
  %res = ptr.ptradd %ptr, %off : !ptr.ptr<#ptr.int_space>, index
  %res0 = ptr.ptradd none %ptr, %off : !ptr.ptr<#ptr.int_space>, index
  %res1 = ptr.ptradd nusw %ptr, %off : !ptr.ptr<#ptr.int_space>, index
  %res2 = ptr.ptradd nuw %ptr, %off : !ptr.ptr<#ptr.int_space>, index
  %res3 = ptr.ptradd inbounds %ptr, %off : !ptr.ptr<#ptr.int_space>, index
  return %res : !ptr.ptr<#ptr.int_space>
}
