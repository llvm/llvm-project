; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s

; CHECK-LABEL: polly.merge_new_and_old:
; CHECK-NEXT: %tmp7.ph.merge = phi ptr [ %tmp7.ph.final_reload, %polly.exiting ], [ %tmp7.ph, %bb6.region_exiting ]

; CHECK-LABEL: polly.stmt.bb3:
; CHECK-NEXT: store ptr %tmp2, ptr %tmp7.s2a

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.blam = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.wibble = type { i32, ptr, ptr }

@global = external global ptr, align 8

; Function Attrs: nounwind uwtable
define void @wobble() #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %tmp2 = phi ptr [ %tmp7, %bb6 ], [ poison, %bb ]
  %tmp = load ptr, ptr @global, align 8, !tbaa !1
  br label %bb3

bb3:                                              ; preds = %bb1
  %tmp4 = getelementptr inbounds %struct.blam, ptr %tmp, i64 0, i32 1
  br i1 false, label %bb6, label %bb5

bb5:                                              ; preds = %bb3
  br label %bb6

bb6:                                              ; preds = %bb5, %bb3
  %tmp7 = phi ptr [ %tmp2, %bb3 ], [ poison, %bb5 ]
  br i1 undef, label %bb8, label %bb1

bb8:                                              ; preds = %bb6
  br label %bb9

bb9:                                              ; preds = %bb8
  unreachable
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 250010) (llvm/trunk 250018)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
