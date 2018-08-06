; RUN: opt -S -mtriple=amdgcn-- -amdgpu-codegenprepare %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: opt -S -mtriple=amdgcn-- -mcpu=tonga -amdgpu-codegenprepare %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: @add_i3(
; SI: %r = add i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @add_i3(i3 %a, i3 %b) {
  %r = add i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nsw_i3(
; SI: %r = add nsw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @add_nsw_i3(i3 %a, i3 %b) {
  %r = add nsw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nuw_i3(
; SI: %r = add nuw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @add_nuw_i3(i3 %a, i3 %b) {
  %r = add nuw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nuw_nsw_i3(
; SI: %r = add nuw nsw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @add_nuw_nsw_i3(i3 %a, i3 %b) {
  %r = add nuw nsw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_i3(
; SI: %r = sub i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @sub_i3(i3 %a, i3 %b) {
  %r = sub i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nsw_i3(
; SI: %r = sub nsw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @sub_nsw_i3(i3 %a, i3 %b) {
  %r = sub nsw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nuw_i3(
; SI: %r = sub nuw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @sub_nuw_i3(i3 %a, i3 %b) {
  %r = sub nuw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nuw_nsw_i3(
; SI: %r = sub nuw nsw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @sub_nuw_nsw_i3(i3 %a, i3 %b) {
  %r = sub nuw nsw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_i3(
; SI: %r = mul i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @mul_i3(i3 %a, i3 %b) {
  %r = mul i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nsw_i3(
; SI: %r = mul nsw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @mul_nsw_i3(i3 %a, i3 %b) {
  %r = mul nsw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nuw_i3(
; SI: %r = mul nuw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @mul_nuw_i3(i3 %a, i3 %b) {
  %r = mul nuw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nuw_nsw_i3(
; SI: %r = mul nuw nsw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @mul_nuw_nsw_i3(i3 %a, i3 %b) {
  %r = mul nuw nsw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_i3(
; SI: %r = shl i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @shl_i3(i3 %a, i3 %b) {
  %r = shl i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nsw_i3(
; SI: %r = shl nsw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @shl_nsw_i3(i3 %a, i3 %b) {
  %r = shl nsw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nuw_i3(
; SI: %r = shl nuw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @shl_nuw_i3(i3 %a, i3 %b) {
  %r = shl nuw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nuw_nsw_i3(
; SI: %r = shl nuw nsw i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @shl_nuw_nsw_i3(i3 %a, i3 %b) {
  %r = shl nuw nsw i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @lshr_i3(
; SI: %r = lshr i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = lshr i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @lshr_i3(i3 %a, i3 %b) {
  %r = lshr i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @lshr_exact_i3(
; SI: %r = lshr exact i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = lshr exact i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @lshr_exact_i3(i3 %a, i3 %b) {
  %r = lshr exact i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @ashr_i3(
; SI: %r = ashr i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = ashr i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @ashr_i3(i3 %a, i3 %b) {
  %r = ashr i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @ashr_exact_i3(
; SI: %r = ashr exact i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = ashr exact i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @ashr_exact_i3(i3 %a, i3 %b) {
  %r = ashr exact i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @and_i3(
; SI: %r = and i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = and i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @and_i3(i3 %a, i3 %b) {
  %r = and i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @or_i3(
; SI: %r = or i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = or i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @or_i3(i3 %a, i3 %b) {
  %r = or i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @xor_i3(
; SI: %r = xor i3 %a, %b
; SI-NEXT: store volatile i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = xor i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @xor_i3(i3 %a, i3 %b) {
  %r = xor i3 %a, %b
  store volatile i3 %r, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_eq_i3(
; SI: %cmp = icmp eq i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp eq i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_eq_i3(i3 %a, i3 %b) {
  %cmp = icmp eq i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ne_i3(
; SI: %cmp = icmp ne i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ne i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_ne_i3(i3 %a, i3 %b) {
  %cmp = icmp ne i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ugt_i3(
; SI: %cmp = icmp ugt i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ugt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_ugt_i3(i3 %a, i3 %b) {
  %cmp = icmp ugt i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_uge_i3(
; SI: %cmp = icmp uge i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp uge i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_uge_i3(i3 %a, i3 %b) {
  %cmp = icmp uge i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ult_i3(
; SI: %cmp = icmp ult i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ult i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_ult_i3(i3 %a, i3 %b) {
  %cmp = icmp ult i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ule_i3(
; SI: %cmp = icmp ule i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ule i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_ule_i3(i3 %a, i3 %b) {
  %cmp = icmp ule i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sgt_i3(
; SI: %cmp = icmp sgt i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sgt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_sgt_i3(i3 %a, i3 %b) {
  %cmp = icmp sgt i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sge_i3(
; SI: %cmp = icmp sge i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sge i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_sge_i3(i3 %a, i3 %b) {
  %cmp = icmp sge i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_slt_i3(
; SI: %cmp = icmp slt i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp slt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_slt_i3(i3 %a, i3 %b) {
  %cmp = icmp slt i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sle_i3(
; SI: %cmp = icmp sle i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: store volatile i3 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sle i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: store volatile i3 %[[SEL_3]]
define amdgpu_kernel void @select_sle_i3(i3 %a, i3 %b) {
  %cmp = icmp sle i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  store volatile i3 %sel, i3 addrspace(1)* undef
  ret void
}

declare i3 @llvm.bitreverse.i3(i3)
; GCN-LABEL: @bitreverse_i3(
; SI: %brev = call i3 @llvm.bitreverse.i3(i3 %a)
; SI-NEXT: store volatile i3 %brev
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[R_32:[0-9]+]] = call i32 @llvm.bitreverse.i32(i32 %[[A_32]])
; VI-NEXT: %[[S_32:[0-9]+]] = lshr i32 %[[R_32]], 29
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[S_32]] to i3
; VI-NEXT: store volatile i3 %[[R_3]]
define amdgpu_kernel void @bitreverse_i3(i3 %a) {
  %brev = call i3 @llvm.bitreverse.i3(i3 %a)
  store volatile i3 %brev, i3 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_i16(
; SI: %r = add i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @add_i16(i16 %a, i16 %b) {
  %r = add i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @constant_add_i16(
; VI: store volatile i16 3
define amdgpu_kernel void @constant_add_i16() {
  %r = add i16 1, 2
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @constant_add_nsw_i16(
; VI: store volatile i16 3
define amdgpu_kernel void @constant_add_nsw_i16() {
  %r = add nsw i16 1, 2
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @constant_add_nuw_i16(
; VI: store volatile i16 3
define amdgpu_kernel void @constant_add_nuw_i16() {
  %r = add nsw i16 1, 2
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nsw_i16(
; SI: %r = add nsw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @add_nsw_i16(i16 %a, i16 %b) {
  %r = add nsw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nuw_i16(
; SI: %r = add nuw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @add_nuw_i16(i16 %a, i16 %b) {
  %r = add nuw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nuw_nsw_i16(
; SI: %r = add nuw nsw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @add_nuw_nsw_i16(i16 %a, i16 %b) {
  %r = add nuw nsw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_i16(
; SI: %r = sub i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @sub_i16(i16 %a, i16 %b) {
  %r = sub i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nsw_i16(
; SI: %r = sub nsw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @sub_nsw_i16(i16 %a, i16 %b) {
  %r = sub nsw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nuw_i16(
; SI: %r = sub nuw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @sub_nuw_i16(i16 %a, i16 %b) {
  %r = sub nuw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nuw_nsw_i16(
; SI: %r = sub nuw nsw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @sub_nuw_nsw_i16(i16 %a, i16 %b) {
  %r = sub nuw nsw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_i16(
; SI: %r = mul i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @mul_i16(i16 %a, i16 %b) {
  %r = mul i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nsw_i16(
; SI: %r = mul nsw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @mul_nsw_i16(i16 %a, i16 %b) {
  %r = mul nsw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nuw_i16(
; SI: %r = mul nuw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @mul_nuw_i16(i16 %a, i16 %b) {
  %r = mul nuw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nuw_nsw_i16(
; SI: %r = mul nuw nsw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @mul_nuw_nsw_i16(i16 %a, i16 %b) {
  %r = mul nuw nsw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_i16(
; SI: %r = shl i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @shl_i16(i16 %a, i16 %b) {
  %r = shl i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nsw_i16(
; SI: %r = shl nsw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @shl_nsw_i16(i16 %a, i16 %b) {
  %r = shl nsw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nuw_i16(
; SI: %r = shl nuw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @shl_nuw_i16(i16 %a, i16 %b) {
  %r = shl nuw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nuw_nsw_i16(
; SI: %r = shl nuw nsw i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @shl_nuw_nsw_i16(i16 %a, i16 %b) {
  %r = shl nuw nsw i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @lshr_i16(
; SI: %r = lshr i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = lshr i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @lshr_i16(i16 %a, i16 %b) {
  %r = lshr i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @lshr_exact_i16(
; SI: %r = lshr exact i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = lshr exact i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @lshr_exact_i16(i16 %a, i16 %b) {
  %r = lshr exact i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @ashr_i16(
; SI: %r = ashr i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = ashr i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @ashr_i16(i16 %a, i16 %b) {
  %r = ashr i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @ashr_exact_i16(
; SI: %r = ashr exact i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = ashr exact i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @ashr_exact_i16(i16 %a, i16 %b) {
  %r = ashr exact i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @constant_lshr_exact_i16(
; VI: store volatile i16 2
define amdgpu_kernel void @constant_lshr_exact_i16(i16 %a, i16 %b) {
  %r = lshr exact i16 4, 1
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @and_i16(
; SI: %r = and i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = and i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @and_i16(i16 %a, i16 %b) {
  %r = and i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @or_i16(
; SI: %r = or i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = or i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @or_i16(i16 %a, i16 %b) {
  %r = or i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @xor_i16(
; SI: %r = xor i16 %a, %b
; SI-NEXT: store volatile i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = xor i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @xor_i16(i16 %a, i16 %b) {
  %r = xor i16 %a, %b
  store volatile i16 %r, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_eq_i16(
; SI: %cmp = icmp eq i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp eq i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_eq_i16(i16 %a, i16 %b) {
  %cmp = icmp eq i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ne_i16(
; SI: %cmp = icmp ne i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ne i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_ne_i16(i16 %a, i16 %b) {
  %cmp = icmp ne i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ugt_i16(
; SI: %cmp = icmp ugt i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ugt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_ugt_i16(i16 %a, i16 %b) {
  %cmp = icmp ugt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_uge_i16(
; SI: %cmp = icmp uge i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp uge i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_uge_i16(i16 %a, i16 %b) {
  %cmp = icmp uge i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ult_i16(
; SI: %cmp = icmp ult i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ult i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_ult_i16(i16 %a, i16 %b) {
  %cmp = icmp ult i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ule_i16(
; SI: %cmp = icmp ule i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ule i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_ule_i16(i16 %a, i16 %b) {
  %cmp = icmp ule i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sgt_i16(
; SI: %cmp = icmp sgt i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sgt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_sgt_i16(i16 %a, i16 %b) {
  %cmp = icmp sgt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sge_i16(
; SI: %cmp = icmp sge i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sge i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_sge_i16(i16 %a, i16 %b) {
  %cmp = icmp sge i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_slt_i16(
; SI: %cmp = icmp slt i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp slt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_slt_i16(i16 %a, i16 %b) {
  %cmp = icmp slt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sle_i16(
; SI: %cmp = icmp sle i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: store volatile i16 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sle i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: store volatile i16 %[[SEL_16]]
define amdgpu_kernel void @select_sle_i16(i16 %a, i16 %b) {
  %cmp = icmp sle i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  store volatile i16 %sel, i16 addrspace(1)* undef
  ret void
}

declare i16 @llvm.bitreverse.i16(i16)

; GCN-LABEL: @bitreverse_i16(
; SI: %brev = call i16 @llvm.bitreverse.i16(i16 %a)
; SI-NEXT: store volatile i16 %brev
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[R_32:[0-9]+]] = call i32 @llvm.bitreverse.i32(i32 %[[A_32]])
; VI-NEXT: %[[S_32:[0-9]+]] = lshr i32 %[[R_32]], 16
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[S_32]] to i16
; VI-NEXT: store volatile i16 %[[R_16]]
define amdgpu_kernel void @bitreverse_i16(i16 %a) {
  %brev = call i16 @llvm.bitreverse.i16(i16 %a)
  store volatile i16 %brev, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_3xi15(
; SI: %r = add <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @add_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = add <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nsw_3xi15(
; SI: %r = add nsw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @add_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = add nsw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nuw_3xi15(
; SI: %r = add nuw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @add_nuw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = add nuw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nuw_nsw_3xi15(
; SI: %r = add nuw nsw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @add_nuw_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = add nuw nsw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_3xi15(
; SI: %r = sub <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @sub_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = sub <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nsw_3xi15(
; SI: %r = sub nsw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @sub_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = sub nsw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nuw_3xi15(
; SI: %r = sub nuw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @sub_nuw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = sub nuw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nuw_nsw_3xi15(
; SI: %r = sub nuw nsw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @sub_nuw_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = sub nuw nsw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_3xi15(
; SI: %r = mul <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @mul_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = mul <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nsw_3xi15(
; SI: %r = mul nsw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @mul_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = mul nsw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nuw_3xi15(
; SI: %r = mul nuw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @mul_nuw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = mul nuw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nuw_nsw_3xi15(
; SI: %r = mul nuw nsw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @mul_nuw_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = mul nuw nsw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_3xi15(
; SI: %r = shl <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @shl_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = shl <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nsw_3xi15(
; SI: %r = shl nsw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @shl_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = shl nsw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nuw_3xi15(
; SI: %r = shl nuw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @shl_nuw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = shl nuw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nuw_nsw_3xi15(
; SI: %r = shl nuw nsw <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @shl_nuw_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = shl nuw nsw <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @lshr_3xi15(
; SI: %r = lshr <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = lshr <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @lshr_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = lshr <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @lshr_exact_3xi15(
; SI: %r = lshr exact <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = lshr exact <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @lshr_exact_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = lshr exact <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @ashr_3xi15(
; SI: %r = ashr <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = ashr <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @ashr_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = ashr <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @ashr_exact_3xi15(
; SI: %r = ashr exact <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = ashr exact <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @ashr_exact_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = ashr exact <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @and_3xi15(
; SI: %r = and <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = and <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @and_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = and <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @or_3xi15(
; SI: %r = or <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = or <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @or_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = or <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @xor_3xi15(
; SI: %r = xor <3 x i15> %a, %b
; SI-NEXT: store volatile <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = xor <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @xor_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = xor <3 x i15> %a, %b
  store volatile <3 x i15> %r, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_eq_3xi15(
; SI: %cmp = icmp eq <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp eq <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_eq_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp eq <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ne_3xi15(
; SI: %cmp = icmp ne <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ne <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_ne_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp ne <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ugt_3xi15(
; SI: %cmp = icmp ugt <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ugt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_ugt_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp ugt <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_uge_3xi15(
; SI: %cmp = icmp uge <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp uge <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_uge_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp uge <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ult_3xi15(
; SI: %cmp = icmp ult <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ult <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_ult_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp ult <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ule_3xi15(
; SI: %cmp = icmp ule <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ule <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_ule_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp ule <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sgt_3xi15(
; SI: %cmp = icmp sgt <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sgt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_sgt_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp sgt <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sge_3xi15(
; SI: %cmp = icmp sge <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sge <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_sge_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp sge <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_slt_3xi15(
; SI: %cmp = icmp slt <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp slt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_slt_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp slt <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sle_3xi15(
; SI: %cmp = icmp sle <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: store volatile <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sle <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[SEL_15]]
define amdgpu_kernel void @select_sle_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp sle <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  store volatile <3 x i15> %sel, <3 x i15> addrspace(1)* undef
  ret void
}

declare <3 x i15> @llvm.bitreverse.v3i15(<3 x i15>)
; GCN-LABEL: @bitreverse_3xi15(
; SI: %brev = call <3 x i15> @llvm.bitreverse.v3i15(<3 x i15> %a)
; SI-NEXT: store volatile <3 x i15> %brev
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = call <3 x i32> @llvm.bitreverse.v3i32(<3 x i32> %[[A_32]])
; VI-NEXT: %[[S_32:[0-9]+]] = lshr <3 x i32> %[[R_32]], <i32 17, i32 17, i32 17>
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[S_32]] to <3 x i15>
; VI-NEXT: store volatile <3 x i15> %[[R_15]]
define amdgpu_kernel void @bitreverse_3xi15(<3 x i15> %a) {
  %brev = call <3 x i15> @llvm.bitreverse.v3i15(<3 x i15> %a)
  store volatile <3 x i15> %brev, <3 x i15> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_3xi16(
; SI: %r = add <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @add_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = add <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nsw_3xi16(
; SI: %r = add nsw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @add_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = add nsw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nuw_3xi16(
; SI: %r = add nuw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @add_nuw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = add nuw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @add_nuw_nsw_3xi16(
; SI: %r = add nuw nsw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @add_nuw_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = add nuw nsw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_3xi16(
; SI: %r = sub <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @sub_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = sub <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nsw_3xi16(
; SI: %r = sub nsw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @sub_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = sub nsw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nuw_3xi16(
; SI: %r = sub nuw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @sub_nuw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = sub nuw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @sub_nuw_nsw_3xi16(
; SI: %r = sub nuw nsw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @sub_nuw_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = sub nuw nsw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_3xi16(
; SI: %r = mul <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @mul_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = mul <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nsw_3xi16(
; SI: %r = mul nsw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @mul_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = mul nsw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nuw_3xi16(
; SI: %r = mul nuw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @mul_nuw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = mul nuw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @mul_nuw_nsw_3xi16(
; SI: %r = mul nuw nsw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @mul_nuw_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = mul nuw nsw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_3xi16(
; SI: %r = shl <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @shl_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = shl <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nsw_3xi16(
; SI: %r = shl nsw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @shl_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = shl nsw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nuw_3xi16(
; SI: %r = shl nuw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @shl_nuw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = shl nuw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @shl_nuw_nsw_3xi16(
; SI: %r = shl nuw nsw <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @shl_nuw_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = shl nuw nsw <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @lshr_3xi16(
; SI: %r = lshr <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = lshr <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @lshr_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = lshr <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @lshr_exact_3xi16(
; SI: %r = lshr exact <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = lshr exact <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @lshr_exact_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = lshr exact <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @ashr_3xi16(
; SI: %r = ashr <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = ashr <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @ashr_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = ashr <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @ashr_exact_3xi16(
; SI: %r = ashr exact <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = ashr exact <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @ashr_exact_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = ashr exact <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @and_3xi16(
; SI: %r = and <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = and <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @and_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = and <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @or_3xi16(
; SI: %r = or <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = or <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @or_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = or <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @xor_3xi16(
; SI: %r = xor <3 x i16> %a, %b
; SI-NEXT: store volatile <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = xor <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @xor_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = xor <3 x i16> %a, %b
  store volatile <3 x i16> %r, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_eq_3xi16(
; SI: %cmp = icmp eq <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp eq <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_eq_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp eq <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ne_3xi16(
; SI: %cmp = icmp ne <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ne <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_ne_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp ne <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ugt_3xi16(
; SI: %cmp = icmp ugt <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ugt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_ugt_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp ugt <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_uge_3xi16(
; SI: %cmp = icmp uge <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp uge <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_uge_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp uge <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ult_3xi16(
; SI: %cmp = icmp ult <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ult <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_ult_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp ult <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_ule_3xi16(
; SI: %cmp = icmp ule <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ule <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_ule_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp ule <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sgt_3xi16(
; SI: %cmp = icmp sgt <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sgt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_sgt_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp sgt <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sge_3xi16(
; SI: %cmp = icmp sge <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sge <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_sge_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp sge <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_slt_3xi16(
; SI: %cmp = icmp slt <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp slt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_slt_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp slt <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: @select_sle_3xi16(
; SI: %cmp = icmp sle <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: store volatile <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sle <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[SEL_16]]
define amdgpu_kernel void @select_sle_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp sle <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store volatile <3 x i16> %sel, <3 x i16> addrspace(1)* undef
  ret void
}

declare <3 x i16> @llvm.bitreverse.v3i16(<3 x i16>)

; GCN-LABEL: @bitreverse_3xi16(
; SI: %brev = call <3 x i16> @llvm.bitreverse.v3i16(<3 x i16> %a)
; SI-NEXT: store volatile <3 x i16> %brev
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = call <3 x i32> @llvm.bitreverse.v3i32(<3 x i32> %[[A_32]])
; VI-NEXT: %[[S_32:[0-9]+]] = lshr <3 x i32> %[[R_32]], <i32 16, i32 16, i32 16>
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[S_32]] to <3 x i16>
; VI-NEXT: store volatile <3 x i16> %[[R_16]]
define amdgpu_kernel void @bitreverse_3xi16(<3 x i16> %a) {
  %brev = call <3 x i16> @llvm.bitreverse.v3i16(<3 x i16> %a)
  store volatile <3 x i16> %brev, <3 x i16> addrspace(1)* undef
  ret void
}
