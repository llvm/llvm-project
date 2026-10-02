; RUN: opt -S -passes='select-function<fn=caller>' < %s | FileCheck %s

; External declarations used by the target should be preserved.
; Unused declarations should be stripped.

; CHECK: declare i32 @extern_used(i32)
declare i32 @extern_used(i32)

; CHECK-NOT: declare {{.*}} @extern_unused
declare i32 @extern_unused(i32)

; CHECK: define {{.*}} @caller(
define i32 @caller(i32 %x) {
  %r = call i32 @extern_used(i32 %x)
  ret i32 %r
}

; CHECK-NOT: @other
define i32 @other(i32 %x) {
  %r = call i32 @extern_unused(i32 %x)
  ret i32 %r
}
