; A call to a libcall function is not a statepoint.
; This test verifies that calls to libcalls functions do not get converted to
; statepoint calls.
; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s

declare double @ldexp(double %x, i32 %n) nounwind readnone

define double @test_libcall(double %x) gc "statepoint-example" {
; CHECK-LABEL: test_libcall
; CHECK-NEXT: %res = call double @ldexp(double %x, i32 5)
; CHECK-NEXT: ret double %res
  %res = call double @ldexp(double %x, i32 5) nounwind readnone
  ret double %res
}
