; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
; Verify we do not crash for this test case and additionally check the code
; that we generate for the %tmp PHI node in the non-affine region. This code
; is difficult to generate as some incoming edges are from basic blocks
; from within the region and others from basic blocks from outside of the
; non-affine region. As visible in the CHECK lines, the code we generate
; currently loads from the PHI twice in %polly.stmt.bb2.entry, which is
; something we should avoid.
;
; CHECK: polly.start

; CHECK: polly.stmt.bb2.entry:                             ; preds = %polly.start
; CHECK-NEXT:   %tmp.phiops.reload = load i32, i32* %tmp.phiops
; CHECK-NEXT:   br label %polly.stmt.bb2

; CHECK: polly.stmt.bb2:                                   ; preds = %polly.stmt.bb2, %polly.stmt.bb2.entry
; CHECK-NEXT:   %polly.tmp = phi i32 [ %tmp.phiops.reload, %polly.stmt.bb2.entry ], [ %p_tmp4, %polly.stmt.bb2 ]
; CHECK-NEXT:   %p_tmp3 = or i32 undef, undef
; CHECK-NEXT:   %p_tmp4 = udiv i32 %p_tmp3, 10
; CHECK-NEXT:   %p_tmp6 = icmp eq i8 undef, 0
; CHECK-NEXT:   br i1 %p_tmp6, label %polly.stmt.polly.merge_new_and_old.exit, label %polly.stmt.bb2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @baz(i32 %before) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  br label %bb2

bb2:                                              ; preds = %bb8, %bb7, %bb2, %bb1
  %tmp = phi i32 [ %before, %bb1 ], [ 0, %bb8 ], [ %tmp4, %bb7 ], [ %tmp4, %bb2 ]
  %tmp3 = or i32 undef, undef
  %tmp4 = udiv i32 %tmp3, 10
  %tmp5 = trunc i32 undef to i8
  %tmp6 = icmp eq i8 %tmp5, 0
  br i1 %tmp6, label %bb7, label %bb2

bb7:                                              ; preds = %bb2
  br i1 undef, label %bb8, label %bb2

bb8:                                              ; preds = %bb7
  br i1 undef, label %bb9, label %bb2

bb9:                                              ; preds = %bb8
  unreachable
}
