; Metal/AIR has no call stack: the helper must be inlined into @kernel and then
; erased, since AIR kernels cannot call defined functions.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK: define void @kernel
; CHECK-NOT: call i32 @helper
; CHECK: ret void
; CHECK-NOT: define {{.*}}@helper

define internal i32 @helper(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}

define void @kernel(ptr %out) {
entry:
  %v = call i32 @helper(i32 41)
  store i32 %v, ptr %out
  ret void
}
