; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -O2 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -O2 | %ptxas-verify %}

; *************************************
; * Cases with no min/max

define i32 @ab_eq_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_eq_i32
; CHECK-NOT: min
; CHECK-NOT: max
  %cmp = icmp eq i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

define i64 @ab_ne_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_ne_i64
; CHECK-NOT: min
; CHECK-NOT: max
  %cmp = icmp ne i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}

; *************************************
; * All variations with i16

; *** ab, unsigned, i16
define i16 @ab_ugt_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ab_ugt_i16
; CHECK: max.u16
  %cmp = icmp ugt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

define i16 @ab_uge_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ab_uge_i16
; CHECK: max.u16
  %cmp = icmp uge i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

define i16 @ab_ult_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ab_ult_i16
; CHECK: min.u16
  %cmp = icmp ult i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

define i16 @ab_ule_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ab_ule_i16
; CHECK: min.u16
  %cmp = icmp ule i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; *** ab, signed, i16
define i16 @ab_sgt_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ab_sgt_i16
; CHECK: max.s16
  %cmp = icmp sgt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

define i16 @ab_sge_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ab_sge_i16
; CHECK: max.s16
  %cmp = icmp sge i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

define i16 @ab_slt_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ab_slt_i16
; CHECK: min.s16
  %cmp = icmp slt i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

define i16 @ab_sle_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ab_sle_i16
; CHECK: min.s16
  %cmp = icmp sle i16 %a, %b
  %sel = select i1 %cmp, i16 %a, i16 %b
  ret i16 %sel
}

; *** ba, unsigned, i16
define i16 @ba_ugt_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ba_ugt_i16
; CHECK: min.u16
  %cmp = icmp ugt i16 %a, %b
  %sel = select i1 %cmp, i16 %b, i16 %a
  ret i16 %sel
}

define i16 @ba_uge_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ba_uge_i16
; CHECK: min.u16
  %cmp = icmp uge i16 %a, %b
  %sel = select i1 %cmp, i16 %b, i16 %a
  ret i16 %sel
}

define i16 @ba_ult_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ba_ult_i16
; CHECK: max.u16
  %cmp = icmp ult i16 %a, %b
  %sel = select i1 %cmp, i16 %b, i16 %a
  ret i16 %sel
}

define i16 @ba_ule_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ba_ule_i16
; CHECK: max.u16
  %cmp = icmp ule i16 %a, %b
  %sel = select i1 %cmp, i16 %b, i16 %a
  ret i16 %sel
}

; *** ba, signed, i16
define i16 @ba_sgt_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ba_sgt_i16
; CHECK: min.s16
  %cmp = icmp sgt i16 %a, %b
  %sel = select i1 %cmp, i16 %b, i16 %a
  ret i16 %sel
}

define i16 @ba_sge_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ba_sge_i16
; CHECK: min.s16
  %cmp = icmp sge i16 %a, %b
  %sel = select i1 %cmp, i16 %b, i16 %a
  ret i16 %sel
}

define i16 @ba_slt_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ba_slt_i16
; CHECK: max.s16
  %cmp = icmp slt i16 %a, %b
  %sel = select i1 %cmp, i16 %b, i16 %a
  ret i16 %sel
}

define i16 @ba_sle_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @ba_sle_i16
; CHECK: max.s16
  %cmp = icmp sle i16 %a, %b
  %sel = select i1 %cmp, i16 %b, i16 %a
  ret i16 %sel
}

; *************************************
; * All variations with i32

; *** ab, unsigned, i32
define i32 @ab_ugt_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_ugt_i32
; CHECK: max.u32
  %cmp = icmp ugt i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

define i32 @ab_uge_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_uge_i32
; CHECK: max.u32
  %cmp = icmp uge i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

define i32 @ab_ult_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_ult_i32
; CHECK: min.u32
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

define i32 @ab_ule_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_ule_i32
; CHECK: min.u32
  %cmp = icmp ule i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

; *** ab, signed, i32
define i32 @ab_sgt_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_sgt_i32
; CHECK: max.s32
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

define i32 @ab_sge_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_sge_i32
; CHECK: max.s32
  %cmp = icmp sge i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

define i32 @ab_slt_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_slt_i32
; CHECK: min.s32
  %cmp = icmp slt i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

define i32 @ab_sle_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ab_sle_i32
; CHECK: min.s32
  %cmp = icmp sle i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

; *** ba, unsigned, i32
define i32 @ba_ugt_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ba_ugt_i32
; CHECK: min.u32
  %cmp = icmp ugt i32 %a, %b
  %sel = select i1 %cmp, i32 %b, i32 %a
  ret i32 %sel
}

define i32 @ba_uge_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ba_uge_i32
; CHECK: min.u32
  %cmp = icmp uge i32 %a, %b
  %sel = select i1 %cmp, i32 %b, i32 %a
  ret i32 %sel
}

define i32 @ba_ult_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ba_ult_i32
; CHECK: max.u32
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, i32 %b, i32 %a
  ret i32 %sel
}

define i32 @ba_ule_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ba_ule_i32
; CHECK: max.u32
  %cmp = icmp ule i32 %a, %b
  %sel = select i1 %cmp, i32 %b, i32 %a
  ret i32 %sel
}

; *** ba, signed, i32
define i32 @ba_sgt_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ba_sgt_i32
; CHECK: min.s32
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, i32 %b, i32 %a
  ret i32 %sel
}

define i32 @ba_sge_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ba_sge_i32
; CHECK: min.s32
  %cmp = icmp sge i32 %a, %b
  %sel = select i1 %cmp, i32 %b, i32 %a
  ret i32 %sel
}

define i32 @ba_slt_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ba_slt_i32
; CHECK: max.s32
  %cmp = icmp slt i32 %a, %b
  %sel = select i1 %cmp, i32 %b, i32 %a
  ret i32 %sel
}

define i32 @ba_sle_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @ba_sle_i32
; CHECK: max.s32
  %cmp = icmp sle i32 %a, %b
  %sel = select i1 %cmp, i32 %b, i32 %a
  ret i32 %sel
}

; *************************************
; * All variations with i64

; *** ab, unsigned, i64
define i64 @ab_ugt_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_ugt_i64
; CHECK: max.u64
  %cmp = icmp ugt i64 %a, %b
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define i64 @ab_uge_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_uge_i64
; CHECK: max.u64
  %cmp = icmp uge i64 %a, %b
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define i64 @ab_ult_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_ult_i64
; CHECK: min.u64
  %cmp = icmp ult i64 %a, %b
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define i64 @ab_ule_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_ule_i64
; CHECK: min.u64
  %cmp = icmp ule i64 %a, %b
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

; *** ab, signed, i64
define i64 @ab_sgt_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_sgt_i64
; CHECK: max.s64
  %cmp = icmp sgt i64 %a, %b
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define i64 @ab_sge_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_sge_i64
; CHECK: max.s64
  %cmp = icmp sge i64 %a, %b
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define i64 @ab_slt_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_slt_i64
; CHECK: min.s64
  %cmp = icmp slt i64 %a, %b
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define i64 @ab_sle_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ab_sle_i64
; CHECK: min.s64
  %cmp = icmp sle i64 %a, %b
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

; *** ba, unsigned, i64
define i64 @ba_ugt_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ba_ugt_i64
; CHECK: min.u64
  %cmp = icmp ugt i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}

define i64 @ba_uge_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ba_uge_i64
; CHECK: min.u64
  %cmp = icmp uge i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}

define i64 @ba_ult_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ba_ult_i64
; CHECK: max.u64
  %cmp = icmp ult i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}

define i64 @ba_ule_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ba_ule_i64
; CHECK: max.u64
  %cmp = icmp ule i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}

; *** ba, signed, i64
define i64 @ba_sgt_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ba_sgt_i64
; CHECK: min.s64
  %cmp = icmp sgt i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}

define i64 @ba_sge_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ba_sge_i64
; CHECK: min.s64
  %cmp = icmp sge i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}

define i64 @ba_slt_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ba_slt_i64
; CHECK: max.s64
  %cmp = icmp slt i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}

define i64 @ba_sle_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @ba_sle_i64
; CHECK: max.s64
  %cmp = icmp sle i64 %a, %b
  %sel = select i1 %cmp, i64 %b, i64 %a
  ret i64 %sel
}
