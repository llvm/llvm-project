; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; Instcombine pulls the addrspacecast out of the select, make sure
;  this doesn't do something insane on non-canonical IR.

; CHECK-LABEL: @return_select_group_flat(
; CHECK-NEXT: %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
; CHECK-NEXT: %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32*
; CHECK-NEXT: %select = select i1 %c, i32* %cast0, i32* %cast1
; CHECK-NEXT: ret i32* %select
define i32* @return_select_group_flat(i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32*
  %select = select i1 %c, i32* %cast0, i32* %cast1
  ret i32* %select
}

; CHECK-LABEL: @store_select_group_flat(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1
; CHECK: store i32 -1, i32 addrspace(3)* %select
define amdgpu_kernel void @store_select_group_flat(i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32*
  %select = select i1 %c, i32* %cast0, i32* %cast1
  store i32 -1, i32* %select
  ret void
}

; Make sure metadata is preserved
; CHECK-LABEL: @load_select_group_flat_md(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1, !prof !0
; CHECK: %load = load i32, i32 addrspace(3)* %select
define i32 @load_select_group_flat_md(i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32*
  %select = select i1 %c, i32* %cast0, i32* %cast1, !prof !0
  %load = load i32, i32* %select
  ret i32 %load
}

; CHECK-LABEL: @store_select_mismatch_group_private_flat(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
; CHECK: %2 = addrspacecast i32 addrspace(5)* %private.ptr.1 to i32*
; CHECK: %select = select i1 %c, i32* %1, i32* %2
; CHECK: store i32 -1, i32* %select
define amdgpu_kernel void @store_select_mismatch_group_private_flat(i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(5)* %private.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %cast1 = addrspacecast i32 addrspace(5)* %private.ptr.1 to i32*
  %select = select i1 %c, i32* %cast0, i32* %cast1
  store i32 -1, i32* %select
  ret void
}

@lds0 = internal addrspace(3) global i32 123, align 4
@lds1 = internal addrspace(3) global i32 456, align 4

; CHECK-LABEL: @constexpr_select_group_flat(
; CHECK: %tmp = load i32, i32 addrspace(3)* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32 addrspace(3)* @lds0, i32 addrspace(3)* @lds1)
define i32 @constexpr_select_group_flat() #0 {
bb:
  %tmp = load i32, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* @lds0 to i32*), i32* addrspacecast (i32 addrspace(3)* @lds1 to i32*))
  ret i32 %tmp
}

; CHECK-LABEL: @constexpr_select_group_global_flat_mismatch(
; CHECK: %tmp = load i32, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* @lds0 to i32*), i32* addrspacecast (i32 addrspace(1)* @global0 to i32*))
define i32 @constexpr_select_group_global_flat_mismatch() #0 {
bb:
  %tmp = load i32, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* @lds0 to i32*), i32* addrspacecast (i32 addrspace(1)* @global0 to i32*))
  ret i32 %tmp
}

; CHECK-LABEL: @store_select_group_flat_null(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* addrspacecast (i32* null to i32 addrspace(3)*)
; CHECK: store i32 -1, i32 addrspace(3)* %select
define amdgpu_kernel void @store_select_group_flat_null(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* %cast0, i32* null
  store i32 -1, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_null_swap(
; CHECK: %select = select i1 %c, i32 addrspace(3)* addrspacecast (i32* null to i32 addrspace(3)*), i32 addrspace(3)* %group.ptr.0
; CHECK: store i32 -1, i32 addrspace(3)* %select
define amdgpu_kernel void @store_select_group_flat_null_swap(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* null, i32* %cast0
  store i32 -1, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_undef(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* undef
; CHECK: store i32 -1, i32 addrspace(3)* %select
define amdgpu_kernel void @store_select_group_flat_undef(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* %cast0, i32* undef
  store i32 -1, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_undef_swap(
; CHECK: %select = select i1 %c, i32 addrspace(3)* undef, i32 addrspace(3)* %group.ptr.0
; CHECK: store i32 -1, i32 addrspace(3)* %select
define amdgpu_kernel void @store_select_group_flat_undef_swap(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* undef, i32* %cast0
  store i32 -1, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_gep_group_flat_null(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* addrspacecast (i32* null to i32 addrspace(3)*)
; CHECK: %gep = getelementptr i32, i32 addrspace(3)* %select, i64 16
; CHECK: store i32 -1, i32 addrspace(3)* %gep
define amdgpu_kernel void @store_select_gep_group_flat_null(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* %cast0, i32* null
  %gep = getelementptr i32, i32* %select, i64 16
  store i32 -1, i32* %gep
  ret void
}

@global0 = internal addrspace(1) global i32 123, align 4

; CHECK-LABEL: @store_select_group_flat_constexpr(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* @lds1
; CHECK: store i32 7, i32 addrspace(3)* %select
define amdgpu_kernel void @store_select_group_flat_constexpr(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* %cast0, i32* addrspacecast (i32 addrspace(3)* @lds1 to i32*)
  store i32 7, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_inttoptr_flat(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* addrspacecast (i32* inttoptr (i64 12345 to i32*) to i32 addrspace(3)*)
; CHECK: store i32 7, i32 addrspace(3)* %select
define amdgpu_kernel void @store_select_group_flat_inttoptr_flat(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* %cast0, i32* inttoptr (i64 12345 to i32*)
  store i32 7, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_inttoptr_group(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* inttoptr (i32 400 to i32 addrspace(3)*)
; CHECK-NEXT: store i32 7, i32 addrspace(3)* %select
define amdgpu_kernel void @store_select_group_flat_inttoptr_group(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* %cast0, i32* addrspacecast (i32 addrspace(3)* inttoptr (i32 400 to i32 addrspace(3)*) to i32*)
  store i32 7, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_flat_constexpr(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
; CHECK: %select = select i1 %c, i32* %1, i32* addrspacecast (i32 addrspace(1)* @global0 to i32*)
; CHECK: store i32 7, i32* %select
define amdgpu_kernel void @store_select_group_global_mismatch_flat_constexpr(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* %cast0, i32* addrspacecast (i32 addrspace(1)* @global0 to i32*)
  store i32 7, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_flat_constexpr_swap(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
; CHECK: %select = select i1 %c, i32* addrspacecast (i32 addrspace(1)* @global0 to i32*), i32* %1
; CHECK: store i32 7, i32* %select
define amdgpu_kernel void @store_select_group_global_mismatch_flat_constexpr_swap(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* addrspacecast (i32 addrspace(1)* @global0 to i32*), i32* %cast0
  store i32 7, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_null_null(
; CHECK: %select = select i1 %c, i32* addrspacecast (i32 addrspace(3)* null to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)
; CHECK: store i32 7, i32* %select
define amdgpu_kernel void @store_select_group_global_mismatch_null_null(i1 %c) #0 {
  %select = select i1 %c, i32* addrspacecast (i32 addrspace(3)* null to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)
  store i32 7, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_null_null_constexpr(
; CHECK: store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* null to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)), align 4
define amdgpu_kernel void @store_select_group_global_mismatch_null_null_constexpr() #0 {
  store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* null to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)), align 4
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_gv_null_constexpr(
; CHECK: store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* @lds0 to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)), align 4
define amdgpu_kernel void @store_select_group_global_mismatch_gv_null_constexpr() #0 {
  store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* @lds0 to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)), align 4
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_null_gv_constexpr(
; CHECK: store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* null to i32*), i32* addrspacecast (i32 addrspace(1)* @global0 to i32*)), align 4
define amdgpu_kernel void @store_select_group_global_mismatch_null_gv_constexpr() #0 {
  store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* null to i32*), i32* addrspacecast (i32 addrspace(1)* @global0 to i32*)), align 4
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_inttoptr_null_constexpr(
; CHECK: store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* inttoptr (i64 123 to i32 addrspace(3)*) to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)), align 4
define amdgpu_kernel void @store_select_group_global_mismatch_inttoptr_null_constexpr() #0 {
  store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* inttoptr (i64 123 to i32 addrspace(3)*) to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)), align 4
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_inttoptr_flat_null_constexpr(
; CHECK: store i32 7, i32 addrspace(1)* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32 addrspace(1)* addrspacecast (i32* inttoptr (i64 123 to i32*) to i32 addrspace(1)*), i32 addrspace(1)* null), align 4
define amdgpu_kernel void @store_select_group_global_mismatch_inttoptr_flat_null_constexpr() #0 {
  store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* inttoptr (i64 123 to i32*), i32* addrspacecast (i32 addrspace(1)* null to i32*)), align 4
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_undef_undef_constexpr(
; CHECK: store i32 7, i32 addrspace(3)* null
define amdgpu_kernel void @store_select_group_global_mismatch_undef_undef_constexpr() #0 {
  store i32 7, i32* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32* addrspacecast (i32 addrspace(3)* null to i32*), i32* addrspacecast (i32 addrspace(1)* undef to i32*)), align 4
  ret void
}

@lds2 = external addrspace(3) global [1024 x i32], align 4

; CHECK-LABEL: @store_select_group_constexpr_ptrtoint(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
; CHECK: %select = select i1 %c, i32* %1, i32* addrspacecast (i32 addrspace(1)* inttoptr (i32 add (i32 ptrtoint ([1024 x i32] addrspace(3)* @lds2 to i32), i32 124) to i32 addrspace(1)*) to i32*)
; CHECK: store i32 7, i32* %select
define amdgpu_kernel void @store_select_group_constexpr_ptrtoint(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32*
  %select = select i1 %c, i32* %cast0, i32* addrspacecast (i32 addrspace(1)* inttoptr (i32 add (i32 ptrtoint ([1024 x i32] addrspace(3)* @lds2 to i32), i32 124) to i32 addrspace(1)*) to i32*)
  store i32 7, i32* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_vector(
; CHECK: %cast0 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.0 to <2 x i32*>
; CHECK: %cast1 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.1 to <2 x i32*>
; CHECK: %select = select i1 %c, <2 x i32*> %cast0, <2 x i32*> %cast1
; CHECK: %extract0 = extractelement <2 x i32*> %select, i32 0
; CHECK: %extract1 = extractelement <2 x i32*> %select, i32 1
; CHECK: store i32 -1, i32* %extract0
; CHECK: store i32 -2, i32* %extract1
define amdgpu_kernel void @store_select_group_flat_vector(i1 %c, <2 x i32 addrspace(3)*> %group.ptr.0, <2 x i32 addrspace(3)*> %group.ptr.1) #0 {
  %cast0 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.0 to <2 x i32*>
  %cast1 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.1 to <2 x i32*>
  %select = select i1 %c, <2 x i32*> %cast0, <2 x i32*> %cast1
  %extract0 = extractelement <2 x i32*> %select, i32 0
  %extract1 = extractelement <2 x i32*> %select, i32 1
  store i32 -1, i32* %extract0
  store i32 -2, i32* %extract1
  ret void
}

attributes #0 = { nounwind }

!0 = !{!"branch_weights", i32 2, i32 10}
