; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
; Check that we generate code without crashing.
;
; CHECK: polly.start
;
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; Function Attrs: nounwind uwtable
define void @quux() unnamed_addr #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp = icmp eq i64 0, 0
  br i1 %tmp, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp3 = icmp ult i64 4, 12
  %tmp4 = zext i1 %tmp3 to i64
  %tmp5 = tail call i64 @llvm.expect.i64(i64 %tmp4, i64 0)
  %tmp6 = trunc i64 %tmp5 to i32
  br label %bb7

bb7:                                              ; preds = %bb2, %bb1
  %tmp8 = phi i32 [ undef, %bb2 ], [ 0, %bb1 ]
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.expect.i64(i64, i64) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }
