; RUN: opt %loadPolly -polly-detect -analyze < %s \
; RUN:     | FileCheck %s
;
; CHECK-NOT: Valid Region for Scop:

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @hoge() #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  br i1 undef, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = load i32, i32* undef, align 8, !tbaa !1
  %tmp3 = tail call i32 @widget() #2
  br i1 false, label %bb4, label %bb5

bb4:                                              ; preds = %bb2
  br label %bb8

bb5:                                              ; preds = %bb2
  %tmp6 = sub i32 %tmp, %tmp3
  br label %bb8

bb7:                                              ; preds = %bb1
  br label %bb8

bb8:                                              ; preds = %bb7, %bb5, %bb4
  ret void
}

; Function Attrs: inlinehint nounwind readonly uwtable
declare i32 @widget() #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint nounwind readonly uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readonly }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 252700) (llvm/trunk 252705)"}
!1 = !{!2, !7, i64 8}
!2 = !{!"cli_target_info", !3, i64 0, !6, i64 8, !4, i64 32}
!3 = !{!"long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!"cli_exe_info", !7, i64 0, !8, i64 4, !3, i64 8, !9, i64 16}
!7 = !{!"int", !4, i64 0}
!8 = !{!"short", !4, i64 0}
!9 = !{!"any pointer", !4, i64 0}
