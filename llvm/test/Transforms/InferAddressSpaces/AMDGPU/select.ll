; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; Instcombine pulls the addrspacecast out of the select, make sure
;  this doesn't do something insane on non-canonical IR.

; CHECK-LABEL: @return_select_group_flat(
; CHECK-NEXT: %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK-NEXT: %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
; CHECK-NEXT: %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* %cast1
; CHECK-NEXT: ret i32 addrspace(4)* %select
define i32 addrspace(4)* @return_select_group_flat(i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* %cast1
  ret i32 addrspace(4)* %select
}

; CHECK-LABEL: @store_select_group_flat(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1
; CHECK: store i32 -1, i32 addrspace(3)* %select
define void @store_select_group_flat(i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* %cast1
  store i32 -1, i32 addrspace(4)* %select
  ret void
}

; Make sure metadata is preserved
; CHECK-LABEL: @load_select_group_flat_md(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1, !prof !0
; CHECK: %load = load i32, i32 addrspace(3)* %select
define i32 @load_select_group_flat_md(i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* %group.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cast1 = addrspacecast i32 addrspace(3)* %group.ptr.1 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* %cast1, !prof !0
  %load = load i32, i32 addrspace(4)* %select
  ret i32 %load
}

; CHECK-LABEL: @store_select_mismatch_group_private_flat(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %2 = addrspacecast i32* %private.ptr.1 to i32 addrspace(4)*
; CHECK: %select = select i1 %c, i32 addrspace(4)* %1, i32 addrspace(4)* %2
; CHECK: store i32 -1, i32 addrspace(4)* %select
define void @store_select_mismatch_group_private_flat(i1 %c, i32 addrspace(3)* %group.ptr.0, i32* %private.ptr.1) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %cast1 = addrspacecast i32* %private.ptr.1 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* %cast1
  store i32 -1, i32 addrspace(4)* %select
  ret void
}

@lds0 = internal addrspace(3) global i32 123, align 4
@lds1 = internal addrspace(3) global i32 456, align 4

; CHECK-LABEL: @constexpr_select_group_flat(
; CHCK: %tmp = load i32, i32 addrspace(3)* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32 addrspace(3)* @lds0, i32 addrspace(3)* @lds1)
define i32 @constexpr_select_group_flat() #0 {
bb:
  %tmp = load i32, i32 addrspace(4)* select (i1 icmp eq (i32 ptrtoint (i32 addrspace(3)* @lds1 to i32), i32 4), i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @lds0 to i32 addrspace(4)*), i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @lds1 to i32 addrspace(4)*))
  ret i32 %tmp
}

; FIXME: Should be able to cast the constants
; CHECK-LABEL: @store_select_group_flat_null(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %select = select i1 %c, i32 addrspace(4)* %1, i32 addrspace(4)* null
; CHECK: store i32 -1, i32 addrspace(4)* %select
define void @store_select_group_flat_null(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* null
  store i32 -1, i32 addrspace(4)* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_null_swap(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %select = select i1 %c, i32 addrspace(4)* null, i32 addrspace(4)* %1
; CHECK: store i32 -1, i32 addrspace(4)* %select
define void @store_select_group_flat_null_swap(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* null, i32 addrspace(4)* %cast0
  store i32 -1, i32 addrspace(4)* %select
  ret void
}


; CHECK-LABEL: @store_select_group_flat_undef(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %select = select i1 %c, i32 addrspace(4)* %1, i32 addrspace(4)* undef
; CHECK: store i32 -1, i32 addrspace(4)* %select
define void @store_select_group_flat_undef(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* undef
  store i32 -1, i32 addrspace(4)* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_undef_swap(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %select = select i1 %c, i32 addrspace(4)* undef, i32 addrspace(4)* %1
; CHECK: store i32 -1, i32 addrspace(4)* %select
define void @store_select_group_flat_undef_swap(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* undef, i32 addrspace(4)* %cast0
  store i32 -1, i32 addrspace(4)* %select
  ret void
}

@global0 = internal addrspace(1) global i32 123, align 4

; CHECK-LABEL: @store_select_group_flat_constexpr(
; CHECK: %select = select i1 %c, i32 addrspace(3)* %group.ptr.0, i32 addrspace(3)* @lds1
; CHECK: store i32 7, i32 addrspace(3)* %select
define void @store_select_group_flat_constexpr(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @lds1 to i32 addrspace(4)*)
  store i32 7, i32 addrspace(4)* %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_flat_constexpr(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %select = select i1 %c, i32 addrspace(4)* %1, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @global0 to i32 addrspace(4)*)
; CHECK: store i32 7, i32 addrspace(4)* %select
define void @store_select_group_global_mismatch_flat_constexpr(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* %cast0, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @global0 to i32 addrspace(4)*)
  store i32 7, i32 addrspace(4)* %select
  ret void
}

; CHECK-LABEL: @store_select_group_global_mismatch_flat_constexpr_swap(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
; CHECK: %select = select i1 %c, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @global0 to i32 addrspace(4)*), i32 addrspace(4)* %1
; CHECK: store i32 7, i32 addrspace(4)* %select
define void @store_select_group_global_mismatch_flat_constexpr_swap(i1 %c, i32 addrspace(3)* %group.ptr.0) #0 {
  %cast0 = addrspacecast i32 addrspace(3)* %group.ptr.0 to i32 addrspace(4)*
  %select = select i1 %c, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @global0 to i32 addrspace(4)*), i32 addrspace(4)* %cast0
  store i32 7, i32 addrspace(4)* %select
  ret void
}

; CHECK-LABEL: @store_select_group_flat_vector(
; CHECK: %cast0 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.0 to <2 x i32 addrspace(4)*>
; CHECK: %cast1 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.1 to <2 x i32 addrspace(4)*>
; CHECK: %select = select i1 %c, <2 x i32 addrspace(4)*> %cast0, <2 x i32 addrspace(4)*> %cast1
; CHECK: %extract0 = extractelement <2 x i32 addrspace(4)*> %select, i32 0
; CHECK: %extract1 = extractelement <2 x i32 addrspace(4)*> %select, i32 1
; CHECK: store i32 -1, i32 addrspace(4)* %extract0
; CHECK: store i32 -2, i32 addrspace(4)* %extract1
define void @store_select_group_flat_vector(i1 %c, <2 x i32 addrspace(3)*> %group.ptr.0, <2 x i32 addrspace(3)*> %group.ptr.1) #0 {
  %cast0 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.0 to <2 x i32 addrspace(4)*>
  %cast1 = addrspacecast <2 x i32 addrspace(3)*> %group.ptr.1 to <2 x i32 addrspace(4)*>
  %select = select i1 %c, <2 x i32 addrspace(4)*> %cast0, <2 x i32 addrspace(4)*> %cast1
  %extract0 = extractelement <2 x i32 addrspace(4)*> %select, i32 0
  %extract1 = extractelement <2 x i32 addrspace(4)*> %select, i32 1
  store i32 -1, i32 addrspace(4)* %extract0
  store i32 -2, i32 addrspace(4)* %extract1
  ret void
}
attributes #0 = { nounwind }

!0 = !{!"branch_weights", i32 2, i32 10}
