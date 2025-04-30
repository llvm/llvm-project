; RUN: opt -passes=next-silicon-split-call-sites -S %s | FileCheck %s

declare void @callee()

; CHECK-LABEL: @simple
define void @simple() {
; CHECK: call
  call void @callee()
; CHECK: br label %[[BB1:.*]]
; CHECK: [[BB1]]:
; CHECK: call
  call void @callee()
  ret void
}

; CHECK-LABEL: @three_calls
define void @three_calls() {
; CHECK: call
  call void @callee()
; CHECK: br label %[[BB1:.*]], !nextsilicon ![[MD0:.*]]
; CHECK: [[BB1]]:
; CHECK: call
  call void @callee()
; CHECK: br label %[[BB2:.*]], !nextsilicon ![[MD0]]
; CHECK: [[BB2]]:
; CHECK: call
  call void @callee()
  ret void, !nextsilicon !0
}

declare ptr @fake_personality_function()

; CHECK-LABEL: @invoke_and_call
define void @invoke_and_call() personality ptr @fake_personality_function {
; CHECK: call
  call void @callee()
; CHECK: br label %[[BB1:.*]]
; CHECK: [[BB1]]:
; CHECK: invoke
  invoke void @callee() to label %normal unwind label %exception
normal:
  ret void
exception:
  landingpad ptr cleanup
  ret void
}

; CHECK-LABEL: @intrinsics
define void @intrinsics(float %val) {
; CHECK-NOT: br
  call float @llvm.exp.f32(float %val)
  call float @llvm.exp.f32(float %val)
  ret void
}

declare float @llvm.exp.f32(float)

; CHECK-LABEL: @cond_br
define void @cond_br(i1 %0) {
; CHECK: call
  call void @callee()
; CHECK: br label %[[BB1:.*]], !nextsilicon ![[MD1:.*]]
; CHECK: [[BB1]]:
; CHECK: call
  call void @callee()
  br i1 %0, label %bb1, label %bb2, !nextsilicon !1
bb1:
  ret void
bb2:
  ret void
}

; CHECK-DAG: ![[MD0]] = !{!"branch_counts", i64 42}
; CHECK-DAG: ![[MD1]] = !{!"branch_counts", i64 66}

!0 = !{!"branch_counts", i64 42}
!1 = !{!"branch_counts", i64 42, i64 24}
