; RUN: opt -S -passes='select-function<fn=target>' < %s | FileCheck %s

; Target function calls @helper, which calls @leaf.
; @unrelated is not reachable from @target and should be removed.
; @unused_global is not referenced by anything kept and should be removed.
; @used_global is referenced by @helper and should be kept.

; CHECK: @used_global = {{.*}} global i32 42
; CHECK-NOT: @unused_global
@used_global = global i32 42
@unused_global = global i32 99

; CHECK: define {{.*}} @target(
define i32 @target(i32 %x) {
  %r = call i32 @helper(i32 %x)
  ret i32 %r
}

; CHECK: define {{.*}} @helper(
define i32 @helper(i32 %x) {
  %val = load i32, ptr @used_global
  %sum = add i32 %x, %val
  %r = call i32 @leaf(i32 %sum)
  ret i32 %r
}

; CHECK: define {{.*}} @leaf(
define i32 @leaf(i32 %x) {
  %r = mul i32 %x, 2
  ret i32 %r
}

; CHECK-NOT: @unrelated
define i32 @unrelated(i32 %x) {
  ret i32 %x
}
