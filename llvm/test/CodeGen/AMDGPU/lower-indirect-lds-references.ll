; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; Tests that the LDS lowering pass handles indirect references to LDS GVs; i.e.
; that it lowers to accesses into the generated LDS struct if these references
; are deep in the call graph starting at the kernel.

@lds_item_to_indirectly_load = internal addrspace(3) global ptr poison, align 8

%store_type = type { i32, ptr }
@place_to_store_indirect_caller = internal addrspace(3) global %store_type poison, align 8

define amdgpu_kernel void @offloading_kernel() {
  store ptr @indirectly_load_lds, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @place_to_store_indirect_caller, i32 0), align 8
  call void @call_unknown()
  ret void
}

define void @call_unknown() {
  %1 = alloca ptr, align 8
  %2 = call i32 %1()
  ret void
}

define void @indirectly_load_lds() {
  call void @directly_load_lds()
  ret void
}

define void @directly_load_lds() {
  %2 = load ptr, ptr addrspace(3) @lds_item_to_indirectly_load, align 8
  ret void
}

; CHECK: %[[LDS_STRUCT_TY:.*]] = type { %store_type, ptr }
; CHECK: @[[LDS_STRUCT:.*]] = {{.*}} %[[LDS_STRUCT_TY]] {{.*}} !absolute_symbol

; CHECK: define amdgpu_kernel void @offloading_kernel() {{.*}} {
; CHECK:   store ptr @indirectly_load_lds, {{.*}} @[[LDS_STRUCT]]
; CHECK:   call void @call_unknown()
; CHECK: }

; CHECK: define void @directly_load_lds() {
; CHECK:   load ptr, {{.*}} (%[[LDS_STRUCT_TY]], {{.*}} @[[LDS_STRUCT]], i32 0, i32 1)
; CHECK: }
