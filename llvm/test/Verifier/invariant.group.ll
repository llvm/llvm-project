; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

; CHECK: invariant.group metadata is only for loads and stores
; CHECK-NEXT: alloca
; CHECK-NEXT: invariant.group metadata is only for loads and stores
; CHECK-NEXT: ret void
define void @f() {
  %a = alloca i32, !invariant.group !0
  %b = load i32, ptr %a, !invariant.group !0
  store i32 43, ptr %a, !invariant.group !0
  ret void, !invariant.group !0
}

!0 = !{}
