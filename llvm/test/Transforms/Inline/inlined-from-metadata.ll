; RUN: opt < %s -passes=inline -S | FileCheck %s

; Verify that the inliner propagates !inlined.from metadata on call sites
; targeting functions with dontcall attributes.

declare void @dontcall_err() "dontcall-error"="error msg"
declare void @dontcall_warn() "dontcall-warn"="warning msg"

define internal void @inner_err() {
  call void @dontcall_err(), !srcloc !0
  ret void
}

define void @test_single_level() {
  call void @inner_err(), !srcloc !1
  ret void
}

; CHECK-LABEL: define void @test_single_level()
; CHECK:         call void @dontcall_err(), !srcloc [[SRCLOC_SINGLE:![0-9]+]], !inlined.from [[SINGLE:![0-9]+]]
; CHECK-NOT:     call void @inner_err

define internal void @inner_warn() {
  call void @dontcall_warn(), !srcloc !2
  ret void
}

define internal void @middle_warn() {
  call void @inner_warn(), !srcloc !3
  ret void
}

define void @test_two_levels() {
  call void @middle_warn(), !srcloc !4
  ret void
}

; CHECK-LABEL: define void @test_two_levels()
; CHECK:         call void @dontcall_warn(), !srcloc [[SRCLOC_TWO:![0-9]+]], !inlined.from [[TWO:![0-9]+]]
; CHECK-NOT:     call void @inner_warn
; CHECK-NOT:     call void @middle_warn

define internal void @multi_calls() {
  call void @dontcall_err(), !srcloc !5
  call void @dontcall_warn(), !srcloc !6
  ret void
}

define void @test_multi_calls() {
  call void @multi_calls(), !srcloc !7
  ret void
}

; CHECK-LABEL: define void @test_multi_calls()
; CHECK:         call void @dontcall_err(), !srcloc {{![0-9]+}}, !inlined.from [[MULTI:![0-9]+]]
; CHECK:         call void @dontcall_warn(), !srcloc {{![0-9]+}}, !inlined.from [[MULTI]]
; CHECK-NOT:     call void @multi_calls

declare void @regular_func()

define internal void @has_regular_call() {
  call void @regular_func(), !srcloc !8
  ret void
}

define void @test_no_metadata_on_regular() {
  call void @has_regular_call(), !srcloc !9
  ret void
}

; CHECK-LABEL: define void @test_no_metadata_on_regular()
; CHECK:         call void @regular_func()
; CHECK-NOT:     !inlined.from

define internal void @no_srcloc_inner() {
  call void @dontcall_err()
  ret void
}

define void @test_no_srcloc() {
  call void @no_srcloc_inner(), !srcloc !10
  ret void
}

; CHECK-LABEL: define void @test_no_srcloc()
; CHECK:         call void @dontcall_err()
; CHECK-NOT:     !inlined.from

!0 = !{i64 100}
!1 = !{i64 200}
!2 = !{i64 300}
!3 = !{i64 400}
!4 = !{i64 500}
!5 = !{i64 600}
!6 = !{i64 700}
!7 = !{i64 800}
!8 = !{i64 900}
!9 = !{i64 1000}
!10 = !{i64 1100}

; CHECK-DAG: [[SRCLOC_SINGLE]] = !{i64 100}
; CHECK-DAG: [[SINGLE]] = !{!"inner_err", i64 0, !"test_single_level", i64 200}
; CHECK-DAG: [[SRCLOC_TWO]] = !{i64 300}
; CHECK-DAG: [[TWO]] = !{!"inner_warn", i64 0, !"middle_warn", i64 400, !"test_two_levels", i64 500}
; CHECK-DAG: [[MULTI]] = !{!"multi_calls", i64 0, !"test_multi_calls", i64 800}
