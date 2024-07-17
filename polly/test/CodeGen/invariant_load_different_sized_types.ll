; RUN: opt %loadNPMPolly -passes=polly-codegen -polly-invariant-load-hoisting=true -S \
; RUN: -polly-allow-differing-element-types < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: polly.preload.begin:  ; preds = %polly.split_new_and_old
; CHECK-NEXT:   %polly.access.tmp2 = getelementptr i32, ptr %tmp2, i64 1
; CHECK-NEXT:   %polly.access.tmp2.load = load i32, ptr %polly.access.tmp2, align 1
; CHECK-NEXT:   store i32 %polly.access.tmp2.load, ptr %tmp.preload.s2a


%struct.hoge = type { [4 x i8], i32, i32, i32, i32, i32, [16 x i8], [16 x i8], i64, i64, i64, i64, i64 }

; Function Attrs: nounwind uwtable
define void @widget() #0 {
bb:
  %tmp2 = alloca %struct.hoge, align 1
  br label %bb3

bb3:                                              ; preds = %bb
  %tmp4 = getelementptr inbounds %struct.hoge, ptr %tmp2, i64 0, i32 10
  %tmp5 = add nsw i32 undef, 1
  %tmp6 = getelementptr inbounds %struct.hoge, ptr %tmp2, i64 0, i32 1
  %tmp = load i32, ptr %tmp6, align 1, !tbaa !1
  %tmp7 = icmp slt i32 %tmp, 3
  br i1 %tmp7, label %bb8, label %bb10

bb8:                                              ; preds = %bb3
  %tmp9 = load i64, ptr %tmp4, align 1, !tbaa !7
  br label %bb10

bb10:                                             ; preds = %bb8, %bb3
  %tmp11 = icmp eq i32 %tmp5, 0
  br i1 %tmp11, label %bb13, label %bb12

bb12:                                             ; preds = %bb10
  ret void

bb13:                                             ; preds = %bb10
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 259751) (llvm/trunk 259771)"}
!1 = !{!2, !5, i64 4}
!2 = !{!"itsf_header_tag", !3, i64 0, !5, i64 4, !5, i64 8, !5, i64 12, !5, i64 16, !5, i64 20, !3, i64 24, !3, i64 40, !6, i64 56, !6, i64 64, !6, i64 72, !6, i64 80, !6, i64 88}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!"int", !3, i64 0}
!6 = !{!"long", !3, i64 0}
!7 = !{!2, !6, i64 72}
