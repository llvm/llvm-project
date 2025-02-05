;; Test reduction of:
;;
;;   DST = shl i64 X, Y
;;
;; where Y is in the range [63-32] to:
;;
;;   DST = [0, shl i32 X, (Y - 32)]

; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck %s

; FIXME: This case should be reduced, but SelectionDAG::computeKnownBits() cannot
;        determine the minimum from metadata in this case.  Match current results
;        for now.
define i64 @shl_metadata(i64 noundef %arg0, ptr %arg1.ptr) {
  %shift.amt = load i64, ptr %arg1.ptr, !range !0
  %shl = shl i64 %arg0, %shift.amt
  ret i64 %shl

; CHECK: .globl  shl_metadata
; CHECK: v_lshl_b64 v[0:1], v[0:1], v2
}

!0 = !{i64 32, i64 64}

; This case is reduced because computeKnownBits() can calculates a minimum of 32
; based on the OR with 32.
define i64 @shl_or32(i64 noundef %arg0, ptr %arg1.ptr) {
  %shift.amt = load i64, ptr %arg1.ptr
  %or = or i64 %shift.amt, 32
  %shl = shl i64 %arg0, %or
  ret i64 %shl

; CHECK: .globl  shl_or32
; CHECK: v_or_b32_e32 v1, 32, v1
; CHECK: v_subrev_i32_e32 v1, vcc, 32, v1
; CHECK: v_lshlrev_b32_e32 v1, v1, v0
; CHECK: v_mov_b32_e32 v0, 0
}

; This case must not be reduced because the known minimum, 16, is not in range.
define i64 @shl_or16(i64 noundef %arg0, ptr %arg1.ptr) {
  %shift.amt = load i64, ptr %arg1.ptr
  %or = or i64 %shift.amt, 16
  %shl = shl i64 %arg0, %or
  ret i64 %shl

; CHECK: .globl  shl_or16
; CHECK: v_or_b32_e32 v2, 16, v2
; CHECK: v_lshl_b64 v[0:1], v[0:1], v2
}

; FIXME: This case should be reduced too, but computeKnownBits() cannot
;        determine the range.  Match current results for now.
define i64 @shl_maxmin(i64 noundef %arg0, i64 noundef %arg1) {
  %max = call i64 @llvm.umax.i64(i64 %arg1, i64 32)
  %min = call i64 @llvm.umin.i64(i64 %max,  i64 63)
  %shl = shl i64 %arg0, %min
  ret i64 %shl

; CHECK: .globl  shl_maxmin
; CHECK: v_cmp_lt_u64_e32 vcc, 32, v[2:3]
; CHECK: v_cndmask_b32_e32 v3, 0, v3, vcc
; CHECK: v_cndmask_b32_e32 v2, 32, v2, vcc
; CHECK: v_cmp_gt_u64_e32 vcc, 63, v[2:3]
; CHECK: v_cndmask_b32_e32 v2, 63, v2, vcc
; CHECK: v_lshl_b64 v[0:1], v[0:1], v2
}
