; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

; Instcombine pulls the addrspacecast out of the select, make sure
;  this doesn't do something insane on non-canonical IR.

; CHECK-LABEL: @return_select_group_flat(
; CHECK-NEXT: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK-NEXT: %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
; CHECK-NEXT: %select = select i1 %c, ptr %cast0, ptr %cast1
; CHECK-NEXT: ret ptr %select
define ptr @return_select_group_flat(i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) %group.ptr.1) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
  %select = select i1 %c, ptr %cast0, ptr %cast1
  ret ptr %select
}

; CHECK-LABEL: @store_select_group_flat(
; CHECK: %select = select i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) %group.ptr.1
; CHECK: store i32 -1, ptr addrspace(3) %select
define amdgpu_kernel void @store_select_group_flat(i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) %group.ptr.1) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
  %select = select i1 %c, ptr %cast0, ptr %cast1
  store i32 -1, ptr %select
  ret void
}

; Make sure metadata is preserved
; CHECK-LABEL: @load_select_group_flat_md(
; CHECK: %select = select i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) %group.ptr.1, !prof !0
; CHECK: %load = load i32, ptr addrspace(3) %select
define i32 @load_select_group_flat_md(i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) %group.ptr.1) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
  %select = select i1 %c, ptr %cast0, ptr %cast1, !prof !0
  %load = load i32, ptr %select
  ret i32 %load
}

; CHECK-LABEL: @store_select_mismatch_group_private_flat(
; CHECK: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK: %cast1 = addrspacecast ptr addrspace(5) %private.ptr.1 to ptr
; CHECK: %select = select i1 %c, ptr %cast0, ptr %cast1
; CHECK: store i32 -1, ptr %select
define amdgpu_kernel void @store_select_mismatch_group_private_flat(i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(5) %private.ptr.1) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cast1 = addrspacecast ptr addrspace(5) %private.ptr.1 to ptr
  %select = select i1 %c, ptr %cast0, ptr %cast1
  store i32 -1, ptr %select
  ret void
}

@lds0 = internal addrspace(3) global i32 123, align 4
@lds1 = internal addrspace(3) global i32 456, align 4

; CHECK-LABEL: @store_select_group_flat_null(
; CHECK: %select = select i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3))
; CHECK: store i32 -1, ptr addrspace(3) %select
define amdgpu_kernel void @store_select_group_flat_null(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr %cast0, ptr null
  store i32 -1, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_null_swap(
; CHECK: %select = select i1 %c, ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr addrspace(3) %group.ptr.0
; CHECK: store i32 -1, ptr addrspace(3) %select
define amdgpu_kernel void @store_select_group_flat_null_swap(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr null, ptr %cast0
  store i32 -1, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_undef(
; CHECK: %select = select i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) undef
; CHECK: store i32 -1, ptr addrspace(3) %select
define amdgpu_kernel void @store_select_group_flat_undef(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr %cast0, ptr undef
  store i32 -1, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_undef_swap(
; CHECK: %select = select i1 %c, ptr addrspace(3) undef, ptr addrspace(3) %group.ptr.0
; CHECK: store i32 -1, ptr addrspace(3) %select
define amdgpu_kernel void @store_select_group_flat_undef_swap(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr undef, ptr %cast0
  store i32 -1, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_gep_group_flat_null(
; CHECK: %select = select i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3))
; CHECK: %gep = getelementptr i32, ptr addrspace(3) %select, i64 16
; CHECK: store i32 -1, ptr addrspace(3) %gep
define amdgpu_kernel void @store_select_gep_group_flat_null(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr %cast0, ptr null
  %gep = getelementptr i32, ptr %select, i64 16
  store i32 -1, ptr %gep
  ret void
}

@global0 = internal addrspace(1) global i32 123, align 4

; CHECK-LABEL: @store_select_group_flat_constexpr(
; CHECK: %select = select i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) @lds1
; CHECK: store i32 7, ptr addrspace(3) %select
define amdgpu_kernel void @store_select_group_flat_constexpr(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr %cast0, ptr addrspacecast (ptr addrspace(3) @lds1 to ptr)
  store i32 7, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_inttoptr_flat(
; CHECK: %select = select i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) addrspacecast (ptr inttoptr (i64 12345 to ptr) to ptr addrspace(3))
; CHECK: store i32 7, ptr addrspace(3) %select
define amdgpu_kernel void @store_select_group_flat_inttoptr_flat(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr %cast0, ptr inttoptr (i64 12345 to ptr)
  store i32 7, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_inttoptr_group(
; CHECK: %select = select i1 %c, ptr addrspace(3) %group.ptr.0, ptr addrspace(3) inttoptr (i32 400 to ptr addrspace(3))
; CHECK-NEXT: store i32 7, ptr addrspace(3) %select
define amdgpu_kernel void @store_select_group_flat_inttoptr_group(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr %cast0, ptr addrspacecast (ptr addrspace(3) inttoptr (i32 400 to ptr addrspace(3)) to ptr)
  store i32 7, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_flat_constexpr(
; CHECK: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK: %select = select i1 %c, ptr %cast0, ptr addrspacecast (ptr addrspace(1) @global0 to ptr)
; CHECK: store i32 7, ptr %select
define amdgpu_kernel void @store_select_group_global_mismatch_flat_constexpr(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr %cast0, ptr addrspacecast (ptr addrspace(1) @global0 to ptr)
  store i32 7, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_flat_constexpr_swap(
; CHECK: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK: %select = select i1 %c, ptr addrspacecast (ptr addrspace(1) @global0 to ptr), ptr %cast0
; CHECK: store i32 7, ptr %select
define amdgpu_kernel void @store_select_group_global_mismatch_flat_constexpr_swap(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr addrspacecast (ptr addrspace(1) @global0 to ptr), ptr %cast0
  store i32 7, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_null_null(
; CHECK: %select = select i1 %c, ptr addrspacecast (ptr addrspace(3) null to ptr), ptr addrspacecast (ptr addrspace(1) null to ptr)
; CHECK: store i32 7, ptr %select
define amdgpu_kernel void @store_select_group_global_mismatch_null_null(i1 %c) #0 {
  %select = select i1 %c, ptr addrspacecast (ptr addrspace(3) null to ptr), ptr addrspacecast (ptr addrspace(1) null to ptr)
  store i32 7, ptr %select
  ret void
}

@lds2 = external addrspace(3) global [1024 x i32], align 4

; CHECK-LABEL: @store_select_group_constexpr_ptrtoint(
; CHECK: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK: %select = select i1 %c, ptr %cast0, ptr addrspacecast (ptr addrspace(1) inttoptr (i32 add (i32 ptrtoint (ptr addrspace(3) @lds2 to i32), i32 124) to ptr addrspace(1)) to ptr)
; CHECK: store i32 7, ptr %select
define amdgpu_kernel void @store_select_group_constexpr_ptrtoint(i1 %c, ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %select = select i1 %c, ptr %cast0, ptr addrspacecast (ptr addrspace(1) inttoptr (i32 add (i32 ptrtoint (ptr addrspace(3) @lds2 to i32), i32 124) to ptr addrspace(1)) to ptr)
  store i32 7, ptr %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_vector(
; CHECK: %cast0 = addrspacecast <2 x ptr addrspace(3)> %group.ptr.0 to <2 x ptr>
; CHECK: %cast1 = addrspacecast <2 x ptr addrspace(3)> %group.ptr.1 to <2 x ptr>
; CHECK: %select = select i1 %c, <2 x ptr> %cast0, <2 x ptr> %cast1
; CHECK: %extract0 = extractelement <2 x ptr> %select, i32 0
; CHECK: %extract1 = extractelement <2 x ptr> %select, i32 1
; CHECK: store i32 -1, ptr %extract0
; CHECK: store i32 -2, ptr %extract1
define amdgpu_kernel void @store_select_group_flat_vector(i1 %c, <2 x ptr addrspace(3)> %group.ptr.0, <2 x ptr addrspace(3)> %group.ptr.1) #0 {
  %cast0 = addrspacecast <2 x ptr addrspace(3)> %group.ptr.0 to <2 x ptr>
  %cast1 = addrspacecast <2 x ptr addrspace(3)> %group.ptr.1 to <2 x ptr>
  %select = select i1 %c, <2 x ptr> %cast0, <2 x ptr> %cast1
  %extract0 = extractelement <2 x ptr> %select, i32 0
  %extract1 = extractelement <2 x ptr> %select, i32 1
  store i32 -1, ptr %extract0
  store i32 -2, ptr %extract1
  ret void
}

attributes #0 = { nounwind }

!0 = !{!"branch_weights", i32 2, i32 10}
