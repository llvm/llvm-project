; RUN: opt -pgo-icall-prom -icp-samplepgo -S < %s | FileCheck %s

define i32* @_Z3fooPi(i32* readnone returned) {
  ret i32* %0
}

; CHECK-LABEL: _Z3barPFPiS_E
; CHECK: if.true.direct_targ
; CHECK:   call i32* @_Z3fooPi
define i32* @_Z3barPFPiS_E(i32* (i32*)* nocapture) {
  %2 = tail call i32* %0(i32* null), !prof !33
  ret i32* %2
}

!llvm.module.flags = !{!3}

!3 = !{i32 1, !"ProfileSummary", !4}
!4 = !{!5, !6, !7, !8, !9, !10, !11, !12}
!5 = !{!"ProfileFormat", !"SampleProfile"}
!6 = !{!"TotalCount", i64 0}
!7 = !{!"MaxCount", i64 0}
!8 = !{!"MaxInternalCount", i64 0}
!9 = !{!"MaxFunctionCount", i64 0}
!10 = !{!"NumCounts", i64 1}
!11 = !{!"NumFunctions", i64 1}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !19, !20, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}
!14 = !{i32 10000, i64 0, i32 0}
!15 = !{i32 100000, i64 0, i32 0}
!16 = !{i32 200000, i64 0, i32 0}
!17 = !{i32 300000, i64 0, i32 0}
!18 = !{i32 400000, i64 0, i32 0}
!19 = !{i32 500000, i64 0, i32 0}
!20 = !{i32 600000, i64 0, i32 0}
!21 = !{i32 700000, i64 0, i32 0}
!22 = !{i32 800000, i64 0, i32 0}
!23 = !{i32 900000, i64 0, i32 0}
!24 = !{i32 950000, i64 0, i32 0}
!25 = !{i32 990000, i64 0, i32 0}
!26 = !{i32 999000, i64 0, i32 0}
!27 = !{i32 999900, i64 0, i32 0}
!28 = !{i32 999990, i64 0, i32 0}
!29 = !{i32 999999, i64 0, i32 0}
!33 = !{!"VP", i32 0, i64 100, i64 8400159624858369790, i64 100}
