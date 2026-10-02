; RUN: opt -S -passes='select-function<fn=entry>' < %s | FileCheck %s

; Diamond dependency: entry -> {left, right} -> bottom.
; All four should be kept. @orphan should be removed.

; CHECK: define {{.*}} @entry(
define i32 @entry(i32 %x) {
  %a = call i32 @left(i32 %x)
  %b = call i32 @right(i32 %x)
  %r = add i32 %a, %b
  ret i32 %r
}

; CHECK: define {{.*}} @left(
define i32 @left(i32 %x) {
  %r = call i32 @bottom(i32 %x)
  ret i32 %r
}

; CHECK: define {{.*}} @right(
define i32 @right(i32 %x) {
  %r = call i32 @bottom(i32 %x)
  ret i32 %r
}

; CHECK: define {{.*}} @bottom(
define i32 @bottom(i32 %x) {
  ret i32 %x
}

; CHECK-NOT: @orphan
define i32 @orphan(i32 %x) {
  %r = call i32 @bottom(i32 %x)
  ret i32 %r
}
