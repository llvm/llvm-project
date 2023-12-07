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
; !memprof metadata should be able to support an arbitrary list of string tags.
!1 = !{!2, !"default", !"tag2"}
!2 = !{i64 123, i64 456}
!3 = !{!4, !"cold", !"tag3", !"tag4"}
!4 = !{i64 123, i64 789, i64 678}
!5 = !{i64 123}
!6 = !{i64 456}
; Inlined callsites will have more than one call stack id.
!7 = !{i64 789, i64 678}
