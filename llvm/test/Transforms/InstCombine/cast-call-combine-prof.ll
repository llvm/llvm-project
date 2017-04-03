; RUN: opt -instcombine -inline -S -inline-threshold=0 -hot-callsite-threshold=100 < %s | FileCheck %s
; Checks if VP profile is used for hotness checks in inlining after instcombine
; converted the call to a direct call.

declare void @bar(i16 *)

define void @foo(i16* %a) {
  call void @bar(i16* %a)
  call void @bar(i16* %a)
  ret void
}

; CHECK-LABEL: @test()
; CHECK-NEXT: call void @bar
; CHECK-NEXT: call void @bar
define void @test() {
  call void bitcast (void (i16*)* @foo to void (i8*)*) (i8* null), !prof !0
  ret void
}

!0 = !{!"VP", i32 0, i64 2000, i64 -3913987384944532146, i64 2000}

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 1000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 1000, i32 1}
!13 = !{i32 999000, i64 1000, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
