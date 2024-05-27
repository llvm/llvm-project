; RUN: opt %loadNPMPolly -passes=polly-codegen -disable-output < %s
; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s | not FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"

define void @MAIN__() nounwind {
entry:
  br i1 undef, label %bb6.preheader, label %bb3

bb3:                                              ; preds = %bb3, %entry
  br i1 undef, label %bb6.preheader, label %bb3

bb6.preheader:                                    ; preds = %bb3, %entry
  br i1 undef, label %bb11, label %bb9.preheader

bb9.preheader:                                    ; preds = %bb6.preheader
  br label %bb11

bb11:                                             ; preds = %bb9.preheader, %bb6.preheader
  br label %bb15

bb15:                                             ; preds = %bb15, %bb11
  br i1 undef, label %bb26.loopexit, label %bb15

bb26.loopexit:                                    ; preds = %bb15
  br i1 undef, label %bb31, label %bb29.preheader

bb29.preheader:                                   ; preds = %bb26.loopexit
  br label %bb29

bb29:                                             ; preds = %bb29, %bb29.preheader
  %indvar47 = phi i32 [ 0, %bb29.preheader ], [ %indvar.next48, %bb29 ] ; <i32> [#uses=1]
  %indvar.next48 = add i32 %indvar47, 1           ; <i32> [#uses=2]
  %exitcond50 = icmp eq i32 %indvar.next48, undef ; <i1> [#uses=1]
  br i1 %exitcond50, label %bb31, label %bb29

bb31:                                             ; preds = %bb29, %bb26.loopexit
  %errtot.3 = phi float [ undef, %bb26.loopexit ], [ undef, %bb29 ] ; <float> [#uses=0]
  ret void
}

; CHECK: SCOP:
