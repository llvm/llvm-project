; RUN: opt -S -passes=instcombine < %s | FileCheck -enable-var-scope %s

; Check that instcombine preserves !prof metadata when removing function
; prototype casts.

declare i32 @__gxx_personality_v0(...)
declare void @__cxa_call_unexpected(ptr)
declare void @foo(ptr %a)

; CHECK-LABEL: @test_call()
; CHECK: call void @foo(ptr null), !prof ![[$PROF:[0-9]+]]
define void @test_call() {
  call void @foo (ptr null), !prof !0
  ret void
}

; CHECK-LABEL: @test_invoke()
; CHECK: invoke void @foo(ptr null)
; CHECK-NEXT: to label %done unwind label %lpad, !prof ![[$PROF]]
define void @test_invoke() personality ptr @__gxx_personality_v0 {
  invoke void @foo (ptr null)
          to label %done unwind label %lpad, !prof !0

done:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %ehptr = extractvalue { ptr, i32 } %lp, 0
  tail call void @__cxa_call_unexpected(ptr %ehptr) noreturn nounwind
  unreachable
}

; CHECK: ![[$PROF]] = !{!"VP", i32 0, i64 2000, i64 -3913987384944532146, i64 2000}
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
