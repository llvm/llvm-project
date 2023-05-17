; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
;
; CHECK-NOT: Valid Region for Scop:

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.hoge = type { i32, i32, i32, i32 }

; Function Attrs: nounwind uwtable
define void @widget() #0 {
bb13:
  %tmp1 = alloca %struct.hoge, align 4
  br i1 undef, label %bb14, label %bb19

bb14:                                             ; preds = %bb13
  %tmp = load i32, ptr undef, align 4, !tbaa !1
  call void @quux() #2
  br i1 false, label %bb15, label %bb18

bb15:                                             ; preds = %bb14
  %tmp16 = getelementptr inbounds %struct.hoge, ptr %tmp1, i64 0, i32 1
  %tmp17 = getelementptr inbounds %struct.hoge, ptr %tmp1, i64 0, i32 2
  br label %bb19

bb18:                                             ; preds = %bb14
  br label %bb19

bb19:                                             ; preds = %bb18, %bb15, %bb13
  %tmp20 = phi i32 [ undef, %bb13 ], [ %tmp, %bb15 ], [ %tmp, %bb18 ]
  unreachable

bb21:                                             ; preds = %bb8
  unreachable

bb22:                                             ; preds = %bb8, %bb8, %bb8, %bb8
  br label %bb23

bb23:                                             ; preds = %bb22
  unreachable

bb24:                                             ; preds = %bb8, %bb8
  unreachable

bb25:                                             ; preds = %bb2
  unreachable
}

declare void @quux() #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 252700) (llvm/trunk 252705)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
