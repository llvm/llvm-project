; RUN: opt -S -passes=place-safepoints < %s | FileCheck %s

; Libcalls will not contain a safepoint poll, so check that we insert
; a safepoint in a loop containing a libcall.
declare double @ldexp(double %x, i32 %n) nounwind readnone
define double @test_libcall(double %x) gc "statepoint-example" {
; CHECK-LABEL: test_libcall

entry:
; CHECK: entry
; CHECK-NEXT: call void @do_safepoint
; CHECK-NEXT: br label %loop
  br label %loop

loop:
; CHECK: loop
; CHECK-NEXT: %x_loop = phi double [ %x, %entry ], [ %x_exp, %loop ]
; CHECK-NEXT: %x_exp = call double @ldexp(double %x_loop, i32 5)
; CHECK-NEXT: %done = fcmp ogt double %x_exp, 1.5
; CHECK-NEXT: call void @do_safepoint
  %x_loop = phi double [ %x, %entry ], [ %x_exp, %loop ]
  %x_exp = call double @ldexp(double %x_loop, i32 5) nounwind readnone
  %done = fcmp ogt double %x_exp, 1.5
  br i1 %done, label %end, label %loop
end:
  %x_end = phi double [%x_exp, %loop]
  ret double %x_end
}

; This function is inlined when inserting a poll.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
entry:
  call void @do_safepoint()
  ret void
}
