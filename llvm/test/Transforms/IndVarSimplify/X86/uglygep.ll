; RUN: opt -passes=indvars -S < %s | FileCheck %s
; rdar://8197217

; Indvars should be able to emit a clean GEP here, not an uglygep.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0"

@numf2s = external global i32                     ; <i32*> [#uses=1]
@numf1s = external global i32                     ; <i32*> [#uses=1]
@tds = external global double**                   ; <double***> [#uses=1]

define void @init_td(i32 %tmp7) nounwind {
; CHECK-LABEL: @init_td
; CHECK-NOT: uglygep
entry:
  br label %bb4

bb4:                                              ; preds = %bb3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %tmp9, %bb3 ]   ; <i32> [#uses=3]
  br label %bb

bb:                                               ; preds = %bb4
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %j.0 = phi i32 [ 0, %bb ], [ %tmp6, %bb1 ]      ; <i32> [#uses=3]
  %tmp8 = icmp slt i32 %j.0, %tmp7                ; <i1> [#uses=1]
  br i1 %tmp8, label %bb1, label %bb3

bb1:                                              ; preds = %bb2
  %tmp = load double**, double*** @tds, align 8             ; <double**> [#uses=1]
  %tmp1 = sext i32 %i.0 to i64                    ; <i64> [#uses=1]
  %tmp2 = getelementptr inbounds double*, double** %tmp, i64 %tmp1 ; <double**> [#uses=1]
  %tmp3 = load double*, double** %tmp2, align 1            ; <double*> [#uses=1]
  %tmp6 = add nsw i32 %j.0, 1                     ; <i32> [#uses=1]
  br label %bb2

bb3:                                              ; preds = %bb2
  %tmp9 = add nsw i32 %i.0, 1                     ; <i32> [#uses=1]
  br label %bb4
}
