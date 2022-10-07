; RUN: llc  -mcpu=pwr9 -mtriple=powerpc64-unknown-linux-gnu < %s -verify-machineinstrs | FileCheck %s
; RUN: llc  -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s -verify-machineinstrs | FileCheck %s
; RUN: llc  -mcpu=pwr8 -mtriple=powerpc64-unknown-linux-gnu -disable-ppc-ctrloops < %s -verify-machineinstrs \
; RUN: | FileCheck %s --check-prefix=CHECK-ITIN
; RUN: llc  -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -disable-ppc-ctrloops < %s -verify-machineinstrs \
; RUN: | FileCheck %s --check-prefix=CHECK-ITIN


%0 = type { i32, i32 }

; Function Attrs: norecurse nounwind writeonly
define void @initCombList(ptr nocapture, i32 signext) local_unnamed_addr #0 {
; CHECK-LABEL: initCombList:
; CHECK: addi 4, 4, -8
; CHECK: stwu [[REG:[0-9]+]], 64(3)

; CHECK-ITIN-LABEL: initCombList:
; CHECK-ITIN: stwu [[REG:[0-9]+]], 64(3)
; CHECK-ITIN-NEXT:   addi [[REG2:[0-9]+]], [[REG2]], 8


  %3 = zext i32 %1 to i64
  br i1 undef, label %6, label %4

; <label>:4:                                      ; preds = %2
  store i32 0, ptr undef, align 4, !tbaa !1
  %5 = add nuw nsw i64 0, 1
  br label %6

; <label>:6:                                      ; preds = %4, %2
  %7 = phi i64 [ 0, %2 ], [ %5, %4 ]
  br i1 undef, label %23, label %8

; <label>:8:                                      ; preds = %8, %6
  %9 = phi i64 [ %21, %8 ], [ %7, %6 ]
  %10 = getelementptr inbounds %0, ptr %0, i64 %9, i32 1
  store i32 0, ptr %10, align 4, !tbaa !1
  %11 = add nuw nsw i64 %9, 1
  %12 = getelementptr inbounds %0, ptr %0, i64 %11, i32 1
  store i32 0, ptr %12, align 4, !tbaa !1
  %13 = add nsw i64 %9, 2
  %14 = getelementptr inbounds %0, ptr %0, i64 %13, i32 1
  store i32 0, ptr %14, align 4, !tbaa !1
  %15 = add nsw i64 %9, 3
  %16 = getelementptr inbounds %0, ptr %0, i64 %15, i32 1
  store i32 0, ptr %16, align 4, !tbaa !1
  %17 = add nsw i64 %9, 4
  %18 = getelementptr inbounds %0, ptr %0, i64 %17, i32 1
  store i32 0, ptr %18, align 4, !tbaa !1
  %19 = add nsw i64 %9, 6
  %20 = getelementptr inbounds %0, ptr %0, i64 %19, i32 1
  store i32 0, ptr %20, align 4, !tbaa !1
  %21 = add nsw i64 %9, 8
  %22 = icmp eq i64 %21, %3
  br i1 %22, label %23, label %8, !llvm.loop !6

; <label>:23:                                     ; preds = %8, %6
  ret void
}

attributes #0 = { norecurse nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+htm,+power8-vector,+vsx,-power9-vector" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 8.0.0 "}
!1 = !{!2, !3, i64 4}
!2 = !{!"", !3, i64 0, !3, i64 4}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.isvectorized", i32 1}
