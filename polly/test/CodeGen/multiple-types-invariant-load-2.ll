; RUN: opt %loadPolly -polly-codegen -S \
; RUN: -polly-allow-differing-element-types < %s | FileCheck %s

; CHECK: polly

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @hoge(i8* %arg) #0 {
bb:
  br label %bb3

bb3:                                              ; preds = %bb
  %tmp = load i8, i8* %arg, align 1, !tbaa !1
  br i1 false, label %bb7, label %bb4

bb4:                                              ; preds = %bb3
  %tmp5 = bitcast i8* %arg to i32*
  %tmp6 = load i32, i32* %tmp5, align 4, !tbaa !4
  br label %bb7

bb7:                                              ; preds = %bb4, %bb3
  %tmp8 = phi i8 [ 1, %bb3 ], [ undef, %bb4 ]
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 259751) (llvm/trunk 259869)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}
