;; Regression test for SCCPSolver propagating !implicit.ref metadata from a
;; callee to its caller before erasing a call instruction that IPSCCP has
;; proven dead after replacing its result with a compile-time constant.
; RUN: opt -passes=ipsccp -S %s | FileCheck %s

@str_callee = internal unnamed_addr constant [4 x i8] c"foo\00"
@str_caller = internal unnamed_addr constant [4 x i8] c"bar\00"

;; callee's return value is provably constant, and the call has no side
;; effects, so IPSCCP will fold the call entirely and erase it.
define internal i32 @callee() nounwind readnone willreturn !implicit.ref !0 {
entry:
  ret i32 42
}

;; caller already carries its own !implicit.ref before the call gets folded.
define i32 @caller() !implicit.ref !1 {
; CHECK-LABEL: define i32 @caller()
; CHECK-SAME: !implicit.ref ![[MD_ORIG:[0-9]+]]
; CHECK-SAME: !implicit.ref ![[MD_PROP:[0-9]+]]
; CHECK-NOT: call i32 @callee(
; CHECK: ret i32 42
entry:
  %call = call i32 @callee()
  ret i32 %call
}

!0 = !{ptr @str_callee}
!1 = !{ptr @str_caller}

; CHECK-DAG: ![[MD_ORIG]] = !{ptr @str_caller}
; CHECK-DAG: ![[MD_PROP]] = !{ptr @str_callee}
