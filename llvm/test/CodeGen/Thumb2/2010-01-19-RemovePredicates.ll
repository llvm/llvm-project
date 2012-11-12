; RUN: llc -O3 -relocation-model=pic -mcpu=cortex-a8 -mattr=+thumb2 < %s
;
; This test creates a predicated t2ADDri instruction that is then turned into a t2MOVgpr2gpr instr.
; Test that that the predicate operands are removed properly.
;
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

declare void @etoe53(i16* nocapture, i16* nocapture) nounwind

define void @earith(double* nocapture %value, i32 %icode, double* nocapture %r1, double* nocapture %r2) nounwind {
entry:
  %v = alloca [6 x i16], align 4                  ; <[6 x i16]*> [#uses=1]
  br i1 undef, label %bb2.i, label %bb5

bb2.i:                                            ; preds = %entry
  %0 = bitcast double* %value to i16*             ; <i16*> [#uses=1]
  call  void @etoe53(i16* null, i16* %0) nounwind
  ret void

bb5:                                              ; preds = %entry
  switch i32 %icode, label %bb10 [
    i32 57, label %bb14
    i32 58, label %bb18
    i32 67, label %bb22
    i32 76, label %bb26
    i32 77, label %bb35
  ]

bb10:                                             ; preds = %bb5
  br label %bb46

bb14:                                             ; preds = %bb5
  unreachable

bb18:                                             ; preds = %bb5
  unreachable

bb22:                                             ; preds = %bb5
  unreachable

bb26:                                             ; preds = %bb5
  br label %bb46

bb35:                                             ; preds = %bb5
  unreachable

bb46:                                             ; preds = %bb26, %bb10
  %1 = bitcast double* %value to i16*             ; <i16*> [#uses=1]
  %v47 = getelementptr inbounds [6 x i16]* %v, i32 0, i32 0 ; <i16*> [#uses=1]
  call  void @etoe53(i16* %v47, i16* %1) nounwind
  ret void
}
