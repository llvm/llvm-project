; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

; CHECK-LABEL: @icmp_flat_cmp_self(
; CHECK: %cmp = icmp eq ptr addrspace(3) %group.ptr.0, %group.ptr.0
define i1 @icmp_flat_cmp_self(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr %cast0, %cast0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_flat_flat_from_group(
; CHECK: %cmp = icmp eq ptr addrspace(3) %group.ptr.0, %group.ptr.1
define i1 @icmp_flat_flat_from_group(ptr addrspace(3) %group.ptr.0, ptr addrspace(3) %group.ptr.1) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
  %cmp = icmp eq ptr %cast0, %cast1
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_flat_from_group_private(
; CHECK: %cast0 = addrspacecast ptr addrspace(5) %private.ptr.0 to ptr
; CHECK: %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
; CHECK: %cmp = icmp eq ptr %cast0, %cast1
define i1 @icmp_mismatch_flat_from_group_private(ptr addrspace(5) %private.ptr.0, ptr addrspace(3) %group.ptr.1) #0 {
  %cast0 = addrspacecast ptr addrspace(5) %private.ptr.0 to ptr
  %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
  %cmp = icmp eq ptr %cast0, %cast1
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_flat_group_flat(
; CHECK: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK: %cmp = icmp eq ptr %cast0, %flat.ptr.1
define i1 @icmp_flat_group_flat(ptr addrspace(3) %group.ptr.0, ptr %flat.ptr.1) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr %cast0, %flat.ptr.1
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_flat_flat_group(
; CHECK: %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
; CHECK: %cmp = icmp eq ptr %flat.ptr.0, %cast1
define i1 @icmp_flat_flat_group(ptr %flat.ptr.0, ptr addrspace(3) %group.ptr.1) #0 {
  %cast1 = addrspacecast ptr addrspace(3) %group.ptr.1 to ptr
  %cmp = icmp eq ptr %flat.ptr.0, %cast1
  ret i1 %cmp
}

; Keeping as cmp addrspace(3)* is better
; CHECK-LABEL: @icmp_flat_to_group_cmp(
; CHECK: %cast0 = addrspacecast ptr %flat.ptr.0 to ptr addrspace(3)
; CHECK: %cast1 = addrspacecast ptr %flat.ptr.1 to ptr addrspace(3)
; CHECK: %cmp = icmp eq ptr addrspace(3) %cast0, %cast1
define i1 @icmp_flat_to_group_cmp(ptr %flat.ptr.0, ptr %flat.ptr.1) #0 {
  %cast0 = addrspacecast ptr %flat.ptr.0 to ptr addrspace(3)
  %cast1 = addrspacecast ptr %flat.ptr.1 to ptr addrspace(3)
  %cmp = icmp eq ptr addrspace(3) %cast0, %cast1
  ret i1 %cmp
}

; FIXME: Should be able to ask target about how to constant fold the
; constant cast if this is OK to change if 0 is a valid pointer.

; CHECK-LABEL: @icmp_group_flat_cmp_null(
; CHECK: %cmp = icmp eq ptr addrspace(3) %group.ptr.0, addrspacecast (ptr null to ptr addrspace(3))
define i1 @icmp_group_flat_cmp_null(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr %cast0, null
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_group_flat_cmp_constant_inttoptr(
; CHECK: %cmp = icmp eq ptr addrspace(3) %group.ptr.0, addrspacecast (ptr inttoptr (i64 400 to ptr) to ptr addrspace(3))
define i1 @icmp_group_flat_cmp_constant_inttoptr(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr %cast0, inttoptr (i64 400 to ptr)
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_flat_group_private_cmp_null(
; CHECK: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK: %cmp = icmp eq ptr %cast0, addrspacecast (ptr addrspace(5) null to ptr)
define i1 @icmp_mismatch_flat_group_private_cmp_null(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr %cast0, addrspacecast (ptr addrspace(5) null to ptr)
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_flat_group_private_cmp_undef(
; CHECK: %cmp = icmp eq ptr addrspace(3) %group.ptr.0, undef
define i1 @icmp_mismatch_flat_group_private_cmp_undef(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr %cast0, addrspacecast (ptr addrspace(5) undef to ptr)
  ret i1 %cmp
}

@lds0 = internal addrspace(3) global i32 0, align 4
@global0 = internal addrspace(1) global i32 0, align 4

; CHECK-LABEL: @icmp_mismatch_flat_group_global_cmp_gv(
; CHECK: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK: %cmp = icmp eq ptr %cast0, addrspacecast (ptr addrspace(1) @global0 to ptr)
define i1 @icmp_mismatch_flat_group_global_cmp_gv(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr %cast0, addrspacecast (ptr addrspace(1) @global0 to ptr)
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_group_global_cmp_gv_gv(
; CHECK: %cmp = icmp eq ptr addrspacecast (ptr addrspace(3) @lds0 to ptr), addrspacecast (ptr addrspace(1) @global0 to ptr)
define i1 @icmp_mismatch_group_global_cmp_gv_gv(ptr addrspace(3) %group.ptr.0) #0 {
  %cmp = icmp eq ptr addrspacecast (ptr addrspace(3) @lds0 to ptr), addrspacecast (ptr addrspace(1) @global0 to ptr)
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_group_flat_cmp_undef(
; CHECK: %cmp = icmp eq ptr addrspace(3) %group.ptr.0, undef
define i1 @icmp_group_flat_cmp_undef(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr %cast0, undef
  ret i1 %cmp
}

; Test non-canonical orders
; CHECK-LABEL: @icmp_mismatch_flat_group_private_cmp_null_swap(
; CHECK: %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
; CHECK: %cmp = icmp eq ptr addrspacecast (ptr addrspace(5) null to ptr), %cast0
define i1 @icmp_mismatch_flat_group_private_cmp_null_swap(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr addrspacecast (ptr addrspace(5) null to ptr), %cast0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_group_flat_cmp_undef_swap(
; CHECK: %cmp = icmp eq ptr addrspace(3) undef, %group.ptr.0
define i1 @icmp_group_flat_cmp_undef_swap(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr undef, %cast0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_flat_group_private_cmp_undef_swap(
; CHECK: %cmp = icmp eq ptr addrspace(3) undef, %group.ptr.0
define i1 @icmp_mismatch_flat_group_private_cmp_undef_swap(ptr addrspace(3) %group.ptr.0) #0 {
  %cast0 = addrspacecast ptr addrspace(3) %group.ptr.0 to ptr
  %cmp = icmp eq ptr addrspacecast (ptr addrspace(5) undef to ptr), %cast0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_flat_flat_from_group_vector(
; CHECK: %cmp = icmp eq <2 x ptr addrspace(3)> %group.ptr.0, %group.ptr.1
define <2 x i1> @icmp_flat_flat_from_group_vector(<2 x ptr addrspace(3)> %group.ptr.0, <2 x ptr addrspace(3)> %group.ptr.1) #0 {
  %cast0 = addrspacecast <2 x ptr addrspace(3)> %group.ptr.0 to <2 x ptr>
  %cast1 = addrspacecast <2 x ptr addrspace(3)> %group.ptr.1 to <2 x ptr>
  %cmp = icmp eq <2 x ptr> %cast0, %cast1
  ret <2 x i1> %cmp
}

attributes #0 = { nounwind }
