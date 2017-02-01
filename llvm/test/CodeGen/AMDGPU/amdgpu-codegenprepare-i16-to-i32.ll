; RUN: opt -S -mtriple=amdgcn-- -amdgpu-codegenprepare %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: opt -S -mtriple=amdgcn-- -mcpu=tonga -amdgpu-codegenprepare %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: @add_i3(
; SI: %r = add i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @add_i3(i3 %a, i3 %b) {
  %r = add i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @add_nsw_i3(
; SI: %r = add nsw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @add_nsw_i3(i3 %a, i3 %b) {
  %r = add nsw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @add_nuw_i3(
; SI: %r = add nuw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @add_nuw_i3(i3 %a, i3 %b) {
  %r = add nuw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @add_nuw_nsw_i3(
; SI: %r = add nuw nsw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @add_nuw_nsw_i3(i3 %a, i3 %b) {
  %r = add nuw nsw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @sub_i3(
; SI: %r = sub i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @sub_i3(i3 %a, i3 %b) {
  %r = sub i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @sub_nsw_i3(
; SI: %r = sub nsw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @sub_nsw_i3(i3 %a, i3 %b) {
  %r = sub nsw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @sub_nuw_i3(
; SI: %r = sub nuw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @sub_nuw_i3(i3 %a, i3 %b) {
  %r = sub nuw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @sub_nuw_nsw_i3(
; SI: %r = sub nuw nsw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @sub_nuw_nsw_i3(i3 %a, i3 %b) {
  %r = sub nuw nsw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @mul_i3(
; SI: %r = mul i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @mul_i3(i3 %a, i3 %b) {
  %r = mul i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @mul_nsw_i3(
; SI: %r = mul nsw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @mul_nsw_i3(i3 %a, i3 %b) {
  %r = mul nsw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @mul_nuw_i3(
; SI: %r = mul nuw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @mul_nuw_i3(i3 %a, i3 %b) {
  %r = mul nuw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @mul_nuw_nsw_i3(
; SI: %r = mul nuw nsw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @mul_nuw_nsw_i3(i3 %a, i3 %b) {
  %r = mul nuw nsw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @urem_i3(
; SI: %r = urem i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = urem i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @urem_i3(i3 %a, i3 %b) {
  %r = urem i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @srem_i3(
; SI: %r = srem i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = srem i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @srem_i3(i3 %a, i3 %b) {
  %r = srem i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @shl_i3(
; SI: %r = shl i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @shl_i3(i3 %a, i3 %b) {
  %r = shl i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @shl_nsw_i3(
; SI: %r = shl nsw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @shl_nsw_i3(i3 %a, i3 %b) {
  %r = shl nsw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @shl_nuw_i3(
; SI: %r = shl nuw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @shl_nuw_i3(i3 %a, i3 %b) {
  %r = shl nuw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @shl_nuw_nsw_i3(
; SI: %r = shl nuw nsw i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @shl_nuw_nsw_i3(i3 %a, i3 %b) {
  %r = shl nuw nsw i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @lshr_i3(
; SI: %r = lshr i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = lshr i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @lshr_i3(i3 %a, i3 %b) {
  %r = lshr i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @lshr_exact_i3(
; SI: %r = lshr exact i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = lshr exact i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @lshr_exact_i3(i3 %a, i3 %b) {
  %r = lshr exact i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @ashr_i3(
; SI: %r = ashr i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = ashr i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @ashr_i3(i3 %a, i3 %b) {
  %r = ashr i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @ashr_exact_i3(
; SI: %r = ashr exact i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = ashr exact i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @ashr_exact_i3(i3 %a, i3 %b) {
  %r = ashr exact i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @and_i3(
; SI: %r = and i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = and i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @and_i3(i3 %a, i3 %b) {
  %r = and i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @or_i3(
; SI: %r = or i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = or i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @or_i3(i3 %a, i3 %b) {
  %r = or i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @xor_i3(
; SI: %r = xor i3 %a, %b
; SI-NEXT: ret i3 %r
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = xor i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[R_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @xor_i3(i3 %a, i3 %b) {
  %r = xor i3 %a, %b
  ret i3 %r
}

; GCN-LABEL: @select_eq_i3(
; SI: %cmp = icmp eq i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp eq i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_eq_i3(i3 %a, i3 %b) {
  %cmp = icmp eq i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_ne_i3(
; SI: %cmp = icmp ne i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ne i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_ne_i3(i3 %a, i3 %b) {
  %cmp = icmp ne i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_ugt_i3(
; SI: %cmp = icmp ugt i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ugt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_ugt_i3(i3 %a, i3 %b) {
  %cmp = icmp ugt i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_uge_i3(
; SI: %cmp = icmp uge i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp uge i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_uge_i3(i3 %a, i3 %b) {
  %cmp = icmp uge i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_ult_i3(
; SI: %cmp = icmp ult i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ult i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_ult_i3(i3 %a, i3 %b) {
  %cmp = icmp ult i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_ule_i3(
; SI: %cmp = icmp ule i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ule i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_ule_i3(i3 %a, i3 %b) {
  %cmp = icmp ule i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_sgt_i3(
; SI: %cmp = icmp sgt i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sgt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_sgt_i3(i3 %a, i3 %b) {
  %cmp = icmp sgt i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_sge_i3(
; SI: %cmp = icmp sge i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sge i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_sge_i3(i3 %a, i3 %b) {
  %cmp = icmp sge i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_slt_i3(
; SI: %cmp = icmp slt i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp slt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_slt_i3(i3 %a, i3 %b) {
  %cmp = icmp slt i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

; GCN-LABEL: @select_sle_i3(
; SI: %cmp = icmp sle i3 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i3 %a, i3 %b
; SI-NEXT: ret i3 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sle i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i3 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i3 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_3:[0-9]+]] = trunc i32 %[[SEL_32]] to i3
; VI-NEXT: ret i3 %[[SEL_3]]
define i3 @select_sle_i3(i3 %a, i3 %b) {
  %cmp = icmp sle i3 %a, %b
  %sel = select i1 %cmp, i3 %a, i3 %b
  ret i3 %sel
}

declare i3 @llvm.bitreverse.i3(i3)
; GCN-LABEL: @bitreverse_i3(
; SI: %brev = call i3 @llvm.bitreverse.i3(i3 %a)
; SI-NEXT: ret i3 %brev
; VI: %[[A_32:[0-9]+]] = zext i3 %a to i32
; VI-NEXT: %[[R_32:[0-9]+]] = call i32 @llvm.bitreverse.i32(i32 %[[A_32]])
; VI-NEXT: %[[S_32:[0-9]+]] = lshr i32 %[[R_32]], 29
; VI-NEXT: %[[R_3:[0-9]+]] = trunc i32 %[[S_32]] to i3
; VI-NEXT: ret i3 %[[R_3]]
define i3 @bitreverse_i3(i3 %a) {
  %brev = call i3 @llvm.bitreverse.i3(i3 %a)
  ret i3 %brev
}

; GCN-LABEL: @add_i16(
; SI: %r = add i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @add_i16(i16 %a, i16 %b) {
  %r = add i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @constant_add_i16(
; VI: ret i16 3
define i16 @constant_add_i16() {
  %r = add i16 1, 2
  ret i16 %r
}

; GCN-LABEL: @constant_add_nsw_i16(
; VI: ret i16 3
define i16 @constant_add_nsw_i16() {
  %r = add nsw i16 1, 2
  ret i16 %r
}

; GCN-LABEL: @constant_add_nuw_i16(
; VI: ret i16 3
define i16 @constant_add_nuw_i16() {
  %r = add nsw i16 1, 2
  ret i16 %r
}

; GCN-LABEL: @add_nsw_i16(
; SI: %r = add nsw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @add_nsw_i16(i16 %a, i16 %b) {
  %r = add nsw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @add_nuw_i16(
; SI: %r = add nuw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @add_nuw_i16(i16 %a, i16 %b) {
  %r = add nuw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @add_nuw_nsw_i16(
; SI: %r = add nuw nsw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @add_nuw_nsw_i16(i16 %a, i16 %b) {
  %r = add nuw nsw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @sub_i16(
; SI: %r = sub i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @sub_i16(i16 %a, i16 %b) {
  %r = sub i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @sub_nsw_i16(
; SI: %r = sub nsw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @sub_nsw_i16(i16 %a, i16 %b) {
  %r = sub nsw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @sub_nuw_i16(
; SI: %r = sub nuw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @sub_nuw_i16(i16 %a, i16 %b) {
  %r = sub nuw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @sub_nuw_nsw_i16(
; SI: %r = sub nuw nsw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @sub_nuw_nsw_i16(i16 %a, i16 %b) {
  %r = sub nuw nsw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @mul_i16(
; SI: %r = mul i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @mul_i16(i16 %a, i16 %b) {
  %r = mul i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @mul_nsw_i16(
; SI: %r = mul nsw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @mul_nsw_i16(i16 %a, i16 %b) {
  %r = mul nsw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @mul_nuw_i16(
; SI: %r = mul nuw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @mul_nuw_i16(i16 %a, i16 %b) {
  %r = mul nuw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @mul_nuw_nsw_i16(
; SI: %r = mul nuw nsw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @mul_nuw_nsw_i16(i16 %a, i16 %b) {
  %r = mul nuw nsw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @urem_i16(
; SI: %r = urem i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = urem i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @urem_i16(i16 %a, i16 %b) {
  %r = urem i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @srem_i16(
; SI: %r = srem i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = srem i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @srem_i16(i16 %a, i16 %b) {
  %r = srem i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @shl_i16(
; SI: %r = shl i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @shl_i16(i16 %a, i16 %b) {
  %r = shl i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @shl_nsw_i16(
; SI: %r = shl nsw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @shl_nsw_i16(i16 %a, i16 %b) {
  %r = shl nsw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @shl_nuw_i16(
; SI: %r = shl nuw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @shl_nuw_i16(i16 %a, i16 %b) {
  %r = shl nuw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @shl_nuw_nsw_i16(
; SI: %r = shl nuw nsw i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @shl_nuw_nsw_i16(i16 %a, i16 %b) {
  %r = shl nuw nsw i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @lshr_i16(
; SI: %r = lshr i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = lshr i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @lshr_i16(i16 %a, i16 %b) {
  %r = lshr i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @lshr_exact_i16(
; SI: %r = lshr exact i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = lshr exact i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @lshr_exact_i16(i16 %a, i16 %b) {
  %r = lshr exact i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @ashr_i16(
; SI: %r = ashr i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = ashr i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @ashr_i16(i16 %a, i16 %b) {
  %r = ashr i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @ashr_exact_i16(
; SI: %r = ashr exact i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = ashr exact i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @ashr_exact_i16(i16 %a, i16 %b) {
  %r = ashr exact i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @constant_lshr_exact_i16(
; VI: ret i16 2
define i16 @constant_lshr_exact_i16(i16 %a, i16 %b) {
  %r = lshr exact i16 4, 1
  ret i16 %r
}

; GCN-LABEL: @and_i16(
; SI: %r = and i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = and i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @and_i16(i16 %a, i16 %b) {
  %r = and i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @or_i16(
; SI: %r = or i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = or i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @or_i16(i16 %a, i16 %b) {
  %r = or i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @xor_i16(
; SI: %r = xor i16 %a, %b
; SI-NEXT: ret i16 %r
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[R_32:[0-9]+]] = xor i32 %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[R_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @xor_i16(i16 %a, i16 %b) {
  %r = xor i16 %a, %b
  ret i16 %r
}

; GCN-LABEL: @select_eq_i16(
; SI: %cmp = icmp eq i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp eq i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_eq_i16(i16 %a, i16 %b) {
  %cmp = icmp eq i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_ne_i16(
; SI: %cmp = icmp ne i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ne i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_ne_i16(i16 %a, i16 %b) {
  %cmp = icmp ne i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_ugt_i16(
; SI: %cmp = icmp ugt i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ugt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_ugt_i16(i16 %a, i16 %b) {
  %cmp = icmp ugt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_uge_i16(
; SI: %cmp = icmp uge i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp uge i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_uge_i16(i16 %a, i16 %b) {
  %cmp = icmp uge i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_ult_i16(
; SI: %cmp = icmp ult i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ult i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_ult_i16(i16 %a, i16 %b) {
  %cmp = icmp ult i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_ule_i16(
; SI: %cmp = icmp ule i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ule i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_ule_i16(i16 %a, i16 %b) {
  %cmp = icmp ule i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_sgt_i16(
; SI: %cmp = icmp sgt i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sgt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_sgt_i16(i16 %a, i16 %b) {
  %cmp = icmp sgt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_sge_i16(
; SI: %cmp = icmp sge i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sge i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_sge_i16(i16 %a, i16 %b) {
  %cmp = icmp sge i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_slt_i16(
; SI: %cmp = icmp slt i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp slt i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_slt_i16(i16 %a, i16 %b) {
  %cmp = icmp slt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; GCN-LABEL: @select_sle_i16(
; SI: %cmp = icmp sle i16 %a, %b
; SI-NEXT: %sel = select i1 %cmp, i16 %a, i16 %b
; SI-NEXT: ret i16 %sel
; VI: %[[A_32_0:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sle i32 %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext i16 %a to i32
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext i16 %b to i32
; VI-NEXT: %[[SEL_32:[0-9]+]] = select i1 %[[CMP]], i32 %[[A_32_1]], i32 %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc i32 %[[SEL_32]] to i16
; VI-NEXT: ret i16 %[[SEL_16]]
define i16 @select_sle_i16(i16 %a, i16 %b) {
  %cmp = icmp sle i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

declare i16 @llvm.bitreverse.i16(i16)
; GCN-LABEL: @bitreverse_i16(
; SI: %brev = call i16 @llvm.bitreverse.i16(i16 %a)
; SI-NEXT: ret i16 %brev
; VI: %[[A_32:[0-9]+]] = zext i16 %a to i32
; VI-NEXT: %[[R_32:[0-9]+]] = call i32 @llvm.bitreverse.i32(i32 %[[A_32]])
; VI-NEXT: %[[S_32:[0-9]+]] = lshr i32 %[[R_32]], 16
; VI-NEXT: %[[R_16:[0-9]+]] = trunc i32 %[[S_32]] to i16
; VI-NEXT: ret i16 %[[R_16]]
define i16 @bitreverse_i16(i16 %a) {
  %brev = call i16 @llvm.bitreverse.i16(i16 %a)
  ret i16 %brev
}

; GCN-LABEL: @add_3xi15(
; SI: %r = add <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @add_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = add <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @add_nsw_3xi15(
; SI: %r = add nsw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @add_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = add nsw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @add_nuw_3xi15(
; SI: %r = add nuw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @add_nuw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = add nuw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @add_nuw_nsw_3xi15(
; SI: %r = add nuw nsw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @add_nuw_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = add nuw nsw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @sub_3xi15(
; SI: %r = sub <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @sub_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = sub <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @sub_nsw_3xi15(
; SI: %r = sub nsw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @sub_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = sub nsw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @sub_nuw_3xi15(
; SI: %r = sub nuw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @sub_nuw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = sub nuw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @sub_nuw_nsw_3xi15(
; SI: %r = sub nuw nsw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @sub_nuw_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = sub nuw nsw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @mul_3xi15(
; SI: %r = mul <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @mul_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = mul <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @mul_nsw_3xi15(
; SI: %r = mul nsw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @mul_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = mul nsw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @mul_nuw_3xi15(
; SI: %r = mul nuw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @mul_nuw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = mul nuw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @mul_nuw_nsw_3xi15(
; SI: %r = mul nuw nsw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @mul_nuw_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = mul nuw nsw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @urem_3xi15(
; SI: %r = urem <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = urem <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @urem_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = urem <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @srem_3xi15(
; SI: %r = srem <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = srem <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @srem_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = srem <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @shl_3xi15(
; SI: %r = shl <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @shl_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = shl <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @shl_nsw_3xi15(
; SI: %r = shl nsw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @shl_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = shl nsw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @shl_nuw_3xi15(
; SI: %r = shl nuw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @shl_nuw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = shl nuw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @shl_nuw_nsw_3xi15(
; SI: %r = shl nuw nsw <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @shl_nuw_nsw_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = shl nuw nsw <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @lshr_3xi15(
; SI: %r = lshr <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = lshr <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @lshr_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = lshr <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @lshr_exact_3xi15(
; SI: %r = lshr exact <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = lshr exact <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @lshr_exact_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = lshr exact <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @ashr_3xi15(
; SI: %r = ashr <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = ashr <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @ashr_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = ashr <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @ashr_exact_3xi15(
; SI: %r = ashr exact <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = ashr exact <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @ashr_exact_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = ashr exact <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @and_3xi15(
; SI: %r = and <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = and <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @and_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = and <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @or_3xi15(
; SI: %r = or <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = or <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @or_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = or <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @xor_3xi15(
; SI: %r = xor <3 x i15> %a, %b
; SI-NEXT: ret <3 x i15> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = xor <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @xor_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %r = xor <3 x i15> %a, %b
  ret <3 x i15> %r
}

; GCN-LABEL: @select_eq_3xi15(
; SI: %cmp = icmp eq <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp eq <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_eq_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp eq <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_ne_3xi15(
; SI: %cmp = icmp ne <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ne <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_ne_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp ne <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_ugt_3xi15(
; SI: %cmp = icmp ugt <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ugt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_ugt_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp ugt <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_uge_3xi15(
; SI: %cmp = icmp uge <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp uge <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_uge_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp uge <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_ult_3xi15(
; SI: %cmp = icmp ult <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ult <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_ult_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp ult <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_ule_3xi15(
; SI: %cmp = icmp ule <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ule <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_ule_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp ule <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_sgt_3xi15(
; SI: %cmp = icmp sgt <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sgt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_sgt_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp sgt <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_sge_3xi15(
; SI: %cmp = icmp sge <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sge <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_sge_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp sge <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_slt_3xi15(
; SI: %cmp = icmp slt <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp slt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_slt_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp slt <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

; GCN-LABEL: @select_sle_3xi15(
; SI: %cmp = icmp sle <3 x i15> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
; SI-NEXT: ret <3 x i15> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sle <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i15> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_15:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[SEL_15]]
define <3 x i15> @select_sle_3xi15(<3 x i15> %a, <3 x i15> %b) {
  %cmp = icmp sle <3 x i15> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i15> %a, <3 x i15> %b
  ret <3 x i15> %sel
}

declare <3 x i15> @llvm.bitreverse.v3i15(<3 x i15>)
; GCN-LABEL: @bitreverse_3xi15(
; SI: %brev = call <3 x i15> @llvm.bitreverse.v3i15(<3 x i15> %a)
; SI-NEXT: ret <3 x i15> %brev
; VI: %[[A_32:[0-9]+]] = zext <3 x i15> %a to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = call <3 x i32> @llvm.bitreverse.v3i32(<3 x i32> %[[A_32]])
; VI-NEXT: %[[S_32:[0-9]+]] = lshr <3 x i32> %[[R_32]], <i32 17, i32 17, i32 17>
; VI-NEXT: %[[R_15:[0-9]+]] = trunc <3 x i32> %[[S_32]] to <3 x i15>
; VI-NEXT: ret <3 x i15> %[[R_15]]
define <3 x i15> @bitreverse_3xi15(<3 x i15> %a) {
  %brev = call <3 x i15> @llvm.bitreverse.v3i15(<3 x i15> %a)
  ret <3 x i15> %brev
}

; GCN-LABEL: @add_3xi16(
; SI: %r = add <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @add_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = add <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @add_nsw_3xi16(
; SI: %r = add nsw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @add_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = add nsw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @add_nuw_3xi16(
; SI: %r = add nuw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @add_nuw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = add nuw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @add_nuw_nsw_3xi16(
; SI: %r = add nuw nsw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = add nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @add_nuw_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = add nuw nsw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @sub_3xi16(
; SI: %r = sub <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @sub_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = sub <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @sub_nsw_3xi16(
; SI: %r = sub nsw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @sub_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = sub nsw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @sub_nuw_3xi16(
; SI: %r = sub nuw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @sub_nuw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = sub nuw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @sub_nuw_nsw_3xi16(
; SI: %r = sub nuw nsw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = sub nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @sub_nuw_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = sub nuw nsw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @mul_3xi16(
; SI: %r = mul <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @mul_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = mul <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @mul_nsw_3xi16(
; SI: %r = mul nsw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @mul_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = mul nsw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @mul_nuw_3xi16(
; SI: %r = mul nuw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @mul_nuw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = mul nuw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @mul_nuw_nsw_3xi16(
; SI: %r = mul nuw nsw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = mul nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @mul_nuw_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = mul nuw nsw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @urem_3xi16(
; SI: %r = urem <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = urem <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @urem_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = urem <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @srem_3xi16(
; SI: %r = srem <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = srem <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @srem_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = srem <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @shl_3xi16(
; SI: %r = shl <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @shl_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = shl <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @shl_nsw_3xi16(
; SI: %r = shl nsw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @shl_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = shl nsw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @shl_nuw_3xi16(
; SI: %r = shl nuw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @shl_nuw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = shl nuw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @shl_nuw_nsw_3xi16(
; SI: %r = shl nuw nsw <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = shl nuw nsw <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @shl_nuw_nsw_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = shl nuw nsw <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @lshr_3xi16(
; SI: %r = lshr <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = lshr <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @lshr_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = lshr <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @lshr_exact_3xi16(
; SI: %r = lshr exact <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = lshr exact <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @lshr_exact_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = lshr exact <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @ashr_3xi16(
; SI: %r = ashr <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = ashr <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @ashr_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = ashr <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @ashr_exact_3xi16(
; SI: %r = ashr exact <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = ashr exact <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @ashr_exact_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = ashr exact <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @and_3xi16(
; SI: %r = and <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = and <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @and_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = and <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @or_3xi16(
; SI: %r = or <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = or <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @or_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = or <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @xor_3xi16(
; SI: %r = xor <3 x i16> %a, %b
; SI-NEXT: ret <3 x i16> %r
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = xor <3 x i32> %[[A_32]], %[[B_32]]
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[R_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @xor_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %r = xor <3 x i16> %a, %b
  ret <3 x i16> %r
}

; GCN-LABEL: @select_eq_3xi16(
; SI: %cmp = icmp eq <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp eq <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_eq_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp eq <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_ne_3xi16(
; SI: %cmp = icmp ne <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ne <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_ne_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp ne <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_ugt_3xi16(
; SI: %cmp = icmp ugt <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ugt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_ugt_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp ugt <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_uge_3xi16(
; SI: %cmp = icmp uge <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp uge <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_uge_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp uge <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_ult_3xi16(
; SI: %cmp = icmp ult <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ult <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_ult_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp ult <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_ule_3xi16(
; SI: %cmp = icmp ule <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp ule <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = zext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_ule_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp ule <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_sgt_3xi16(
; SI: %cmp = icmp sgt <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sgt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_sgt_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp sgt <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_sge_3xi16(
; SI: %cmp = icmp sge <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sge <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_sge_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp sge <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_slt_3xi16(
; SI: %cmp = icmp slt <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp slt <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_slt_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp slt <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

; GCN-LABEL: @select_sle_3xi16(
; SI: %cmp = icmp sle <3 x i16> %a, %b
; SI-NEXT: %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
; SI-NEXT: ret <3 x i16> %sel
; VI: %[[A_32_0:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_0:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[CMP:[0-9]+]] = icmp sle <3 x i32> %[[A_32_0]], %[[B_32_0]]
; VI-NEXT: %[[A_32_1:[0-9]+]] = sext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[B_32_1:[0-9]+]] = sext <3 x i16> %b to <3 x i32>
; VI-NEXT: %[[SEL_32:[0-9]+]] = select <3 x i1> %[[CMP]], <3 x i32> %[[A_32_1]], <3 x i32> %[[B_32_1]]
; VI-NEXT: %[[SEL_16:[0-9]+]] = trunc <3 x i32> %[[SEL_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[SEL_16]]
define <3 x i16> @select_sle_3xi16(<3 x i16> %a, <3 x i16> %b) {
  %cmp = icmp sle <3 x i16> %a, %b
  %sel = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  ret <3 x i16> %sel
}

declare <3 x i16> @llvm.bitreverse.v3i16(<3 x i16>)
; GCN-LABEL: @bitreverse_3xi16(
; SI: %brev = call <3 x i16> @llvm.bitreverse.v3i16(<3 x i16> %a)
; SI-NEXT: ret <3 x i16> %brev
; VI: %[[A_32:[0-9]+]] = zext <3 x i16> %a to <3 x i32>
; VI-NEXT: %[[R_32:[0-9]+]] = call <3 x i32> @llvm.bitreverse.v3i32(<3 x i32> %[[A_32]])
; VI-NEXT: %[[S_32:[0-9]+]] = lshr <3 x i32> %[[R_32]], <i32 16, i32 16, i32 16>
; VI-NEXT: %[[R_16:[0-9]+]] = trunc <3 x i32> %[[S_32]] to <3 x i16>
; VI-NEXT: ret <3 x i16> %[[R_16]]
define <3 x i16> @bitreverse_3xi16(<3 x i16> %a) {
  %brev = call <3 x i16> @llvm.bitreverse.v3i16(<3 x i16> %a)
  ret <3 x i16> %brev
}
