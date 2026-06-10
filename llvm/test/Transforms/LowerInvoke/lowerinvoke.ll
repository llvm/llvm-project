; RUN: opt < %s -passes=lower-invoke -S | FileCheck %s
; RUN: opt < %s -passes='lower-invoke' -S | FileCheck %s

declare i32 @external_func(i64 %arg)

define i32 @invoke_test(i64 %arg) personality ptr null {
entry:
  %result = invoke fastcc i32 @external_func(i64 inreg %arg)
      to label %cont unwind label %lpad
cont:
  ret i32 %result
lpad:
  %phi = phi i32 [ 99, %entry ]
  %lp = landingpad { ptr, i32 } cleanup
  ret i32 %phi
}

; The "invoke" should be converted to a "call".
; CHECK-LABEL: define i32 @invoke_test
; CHECK: %result = call fastcc i32 @external_func(i64 inreg %arg)
; CHECK-NEXT: br label %cont

; Note that this pass does not remove dead landingpad blocks.
; CHECK: lpad:
; CHECK-NOT: phi
; CHECK: landingpad

; The lowered call must carry the invoke's metadata (e.g. !noalias), and the
; invoke's two-weight !prof is converted to the call's single total weight.
declare i32 @g()
declare i32 @__gxx_personality_v0(...)
define i32 @keep_md() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @keep_md(
; CHECK: call i32 @g(){{.*}}!prof ![[PROF:[0-9]+]]{{.*}}!noalias
; CHECK: ![[PROF]] = !{!"branch_weights", i32 101}
entry:
  %r = invoke i32 @g() to label %cont unwind label %lpad, !prof !0, !noalias !1
cont:
  ret i32 %r
lpad:
  %l = landingpad { ptr, i32 } cleanup
  ret i32 0
}
!0 = !{!"branch_weights", i32 100, i32 1}
!1 = !{!2}
!2 = distinct !{!2, !3}
!3 = distinct !{!3}
