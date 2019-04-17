; RUN: opt < %s -jump-threading -disable-output -verify-dom-info

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = external local_unnamed_addr global i64, align 8
@global.1 = external local_unnamed_addr global i64, align 8
@global.2 = external local_unnamed_addr global i64, align 8

; Function Attrs: norecurse noreturn nounwind uwtable
define void @hoge() local_unnamed_addr #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb26, %bb
  %tmp = load i64, i64* @global, align 8, !tbaa !1
  %tmp2 = icmp eq i64 %tmp, 0
  br i1 %tmp2, label %bb27, label %bb3

bb3:                                              ; preds = %bb1
  %tmp4 = load i64, i64* @global.1, align 8, !tbaa !1
  %tmp5 = icmp eq i64 %tmp4, 0
  br i1 %tmp5, label %bb23, label %bb23

bb23:                                             ; preds = %bb3, %bb3
  br label %bb26

bb26:                                             ; preds = %bb27, %bb23
  br label %bb1

bb27:                                             ; preds = %bb1
  br label %bb26
}

attributes #0 = { norecurse noreturn nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 7.0.0 "}
!1 = !{!2, !2, i64 0}
!2 = !{!"long", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
