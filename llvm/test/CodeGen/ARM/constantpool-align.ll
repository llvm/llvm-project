; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-arm-none-eabi"

; CHECK-LABEL: f:
; CHECK: vld1.64 {{.*}}, [r1:128]
; CHECK: .p2align 4
define void @f(<4 x i32>* %p) {
  store <4 x i32> <i32 -1, i32 0, i32 0, i32 -1>, <4 x i32>* %p, align 4
  ret void 
}

; CHECK-LABEL: f_optsize:
; CHECK: vld1.64 {{.*}}, [r1]
; CHECK: .p2align 3
define void @f_optsize(<4 x i32>* %p) optsize {
  store <4 x i32> <i32 -1, i32 0, i32 0, i32 -1>, <4 x i32>* %p, align 4
  ret void 
}

; CHECK-LABEL: f_pgso:
; CHECK: vld1.64 {{.*}}, [r1]
; CHECK: .p2align 3
define void @f_pgso(<4 x i32>* %p) !prof !14 {
  store <4 x i32> <i32 -1, i32 0, i32 0, i32 -1>, <4 x i32>* %p, align 4
  ret void 
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
