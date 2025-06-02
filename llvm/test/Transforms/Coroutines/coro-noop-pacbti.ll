
; RUN: opt < %s -S -passes=coro-early | FileCheck %s

; CHECK:      define private fastcc void @__NoopCoro_ResumeDestroy(ptr %0) #1 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK:      attributes #1 = { "branch-target-enforcement" "sign-return-address"="all" "sign-return-address-key"="a_key" }

define ptr @noop() {
entry:
  %n = call ptr @llvm.coro.noop()
  ret ptr %n
}

declare ptr @llvm.coro.noop()

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 1}
