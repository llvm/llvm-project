; RUN: opt -passes=inline < %s -disable-output -debug-pass-manager 2>&1 | FileCheck %s

; We shouldn't invalidate any function analyses on g since it's never modified.

; CHECK-NOT: Invalidating{{.*}} on g
; CHECK: Invalidating{{.*}} on f
; CHECK-NOT: Invalidating{{.*}} on g

define void @f() noinline {
  call void @g()
  ret void
}

define void @g() alwaysinline {
  call void @f()
  ret void
}
