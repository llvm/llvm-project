; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; CHECK-LABEL: @icmp_flat_cmp_self(
; CHECK: %cmp = icmp eq i32 addrspace(3)* %group.ptr.0, %group.ptr.0
define i1 @icmp_flat_cmp_self(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, %cast0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_flat_flat_from_group(
; CHECK: %cmp = icmp eq i32 addrspace(3)* %group.ptr.0, %group.ptr.1
define i1 @icmp_flat_flat_from_group(i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, %cast1
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_flat_from_group_private(
; CHECK: %1 = addrspacecast i32* %private.ptr.0 to i32 addrspace(4)*
; CHECK: %2 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
; CHECK: %cmp = icmp eq i32 addrspace(4)* %1, %2
define i1 @icmp_mismatch_flat_from_group_private(i32* %private.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast0 = addrspacecast i32* %private.ptr.0 to i32 addrspace(4)*
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, %cast1
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_flat_group_flat(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %cmp = icmp eq i32 addrspace(4)* %1, %flat.ptr.1
define i1 @icmp_flat_group_flat(i32 addrspace(3)* %group.ptr.0, i32 addrspace(4)* %flat.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, %flat.ptr.1
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_flat_flat_group(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
; CHECK: %cmp = icmp eq i32 addrspace(4)* %flat.ptr.0, %1
define i1 @icmp_flat_flat_group(i32 addrspace(4)* %flat.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %flat.ptr.0, %cast1
  ret i1 %cmp
}

; Keeping as cmp addrspace(3)* is better
; CHECK-LABEL: @icmp_flat_to_group_cmp(
; CHECK: %cast0 = addrspacecast i32 addrspace(4)* %flat.ptr.0 to i32 addrspace(3)*
; CHECK: %cast1 = addrspacecast i32 addrspace(4)* %flat.ptr.1 to i32 addrspace(3)*
; CHECK: %cmp = icmp eq i32 addrspace(3)* %cast0, %cast1
define i1 @icmp_flat_to_group_cmp(i32 addrspace(4)* %flat.ptr.0, i32 addrspace(4)* %flat.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(4)* %flat.ptr.0 to i32 addrspace(3)*
  %cast1 = addrspacecast i32 addrspace(4)* %flat.ptr.1 to i32 addrspace(3)*
  %cmp = icmp eq i32 addrspace(3)* %cast0, %cast1
  ret i1 %cmp
}

; FIXME: Should be able to ask target about how to constant fold the
; constant cast if this is OK to change if 0 is a valid pointer.

; CHECK-LABEL: @icmp_group_flat_cmp_null(
; CHECK: %cmp = icmp eq i32 addrspace(3)* %group.ptr.0, addrspacecast (i32 addrspace(4)* null to i32 addrspace(3)*)
define i1 @icmp_group_flat_cmp_null(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, null
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_group_flat_cmp_constant_inttoptr(
; CHECK: %cmp = icmp eq i32 addrspace(3)* %group.ptr.0, addrspacecast (i32 addrspace(4)* inttoptr (i64 400 to i32 addrspace(4)*) to i32 addrspace(3)*)
define i1 @icmp_group_flat_cmp_constant_inttoptr(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, inttoptr (i64 400 to i32 addrspace(4)*)
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_flat_group_private_cmp_null(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %cmp = icmp eq i32 addrspace(4)* %1, addrspacecast (i32* null to i32 addrspace(4)*)
define i1 @icmp_mismatch_flat_group_private_cmp_null(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, addrspacecast (i32* null to i32 addrspace(4)*)
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_flat_group_private_cmp_undef(
; CHECK: %cmp = icmp eq i32 addrspace(3)* %group.ptr.0, undef
define i1 @icmp_mismatch_flat_group_private_cmp_undef(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, addrspacecast (i32* undef to i32 addrspace(4)*)
  ret i1 %cmp
}

@lds0 = internal addrspace(3) global i32 0, align 4
@global0 = internal addrspace(1) global i32 0, align 4

; CHECK-LABEL: @icmp_mismatch_flat_group_global_cmp_gv(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %cmp = icmp eq i32 addrspace(4)* %1, addrspacecast (i32 addrspace(1)* @global0 to i32 addrspace(4)*)
define i1 @icmp_mismatch_flat_group_global_cmp_gv(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, addrspacecast (i32 addrspace(1)* @global0 to i32 addrspace(4)*)
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_group_global_cmp_gv_gv(
; CHECK: %cmp = icmp eq i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @lds0 to i32 addrspace(4)*), addrspacecast (i32 addrspace(1)* @global0 to i32 addrspace(4)*)
define i1 @icmp_mismatch_group_global_cmp_gv_gv(i32 addrspace(3)* %group.ptr.0) #0 {
  %cmp = icmp eq i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @lds0 to i32 addrspace(4)*), addrspacecast (i32 addrspace(1)* @global0 to i32 addrspace(4)*)
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_group_flat_cmp_undef(
; CHECK: %cmp = icmp eq i32 addrspace(3)* %group.ptr.0, undef
define i1 @icmp_group_flat_cmp_undef(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* %cast0, undef
  ret i1 %cmp
}

; Test non-canonical orders
; CHECK-LABEL: @icmp_mismatch_flat_group_private_cmp_null_swap(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %cmp = icmp eq i32 addrspace(4)* addrspacecast (i32* null to i32 addrspace(4)*), %1
define i1 @icmp_mismatch_flat_group_private_cmp_null_swap(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* addrspacecast (i32* null to i32 addrspace(4)*), %cast0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_group_flat_cmp_undef_swap(
; CHECK: %cmp = icmp eq i32 addrspace(3)* undef, %group.ptr.0
define i1 @icmp_group_flat_cmp_undef_swap(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* undef, %cast0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mismatch_flat_group_private_cmp_undef_swap(
; CHECK: %cmp = icmp eq i32 addrspace(3)* undef, %group.ptr.0
define i1 @icmp_mismatch_flat_group_private_cmp_undef_swap(i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cmp = icmp eq i32 addrspace(4)* addrspacecast (i32* undef to i32 addrspace(4)*), %cast0
  ret i1 %cmp
}

; TODO: Should be handled
; CHECK-LABEL: @icmp_flat_flat_from_group_vector(
; CHECK: %cmp = icmp eq <2 x i32 addrspace(4)*> %cast0, %cast1
define <2 x i1> @icmp_flat_flat_from_group_vector(<2 x i32 addrspace(3)*> %group.ptr.0, <2 x i32 addrspace(3)*> %group.ptr.1) #0 {
  %cast0 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.0 to <2 x i32 addrspace(4)*>
  %cast1 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.1 to <2 x i32 addrspace(4)*>
  %cmp = icmp eq <2 x i32 addrspace(4)*> %cast0, %cast1
  ret <2 x i1> %cmp
}

attributes #0 = { nounwind }
