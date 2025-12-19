; Test that well-formed memprof and callsite metadata pass verification.
; RUN: llvm-as -disable-output < %s 2>&1

define ptr @test1() {
entry:
  %call = call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40), !memprof !0, !callsite !5
  ret ptr %call
}

define ptr @test2() {
entry:
  %call = call noundef ptr @test1(), !callsite !6
  ret ptr %call
}

define ptr @test3() {
entry:
  %call = call noundef ptr @test1(), !callsite !7
  ret ptr %call
}

declare dso_local noalias noundef ptr @malloc(i64 noundef)

!0 = !{!1, !3}
; !memprof metadata can have a single string tag, followed by an arbitrary
; list of MDNodes. The MDNodes should each be a pair
!1 = !{!2, !"default", !8, !9}
!2 = !{i64 123, i64 456}
!3 = !{!4, !"cold", !10}
!4 = !{i64 123, i64 789, i64 678}
!5 = !{i64 123}
!6 = !{i64 456}
; Inlined callsites will have more than one call stack id.
!7 = !{i64 789, i64 678}
!8 = !{i64 891, i64 100}
!9 = !{i64 912, i64 200}
!10 = !{i64 1234, i64 300}
