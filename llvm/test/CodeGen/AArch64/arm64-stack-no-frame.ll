; RUN: llc -mtriple=arm64-apple-ios7.0 -o - %s | FileCheck %s

@global = global [20 x i64] zeroinitializer, align 8

; The following function has enough locals to need some restoring, but not a
; frame record. In an intermediate frame refactoring, prologue and epilogue were
; inconsistent about how much to move SP.
define void @test_stack_no_frame() {
; CHECK: test_stack_no_frame
; CHECK: sub sp, sp, #[[STACKSIZE:[0-9]+]]
  %local = alloca [20 x i64]
  %val = load volatile [20 x i64], ptr @global, align 8
  store volatile [20 x i64] %val, ptr %local, align 8

  %val2 = load volatile [20 x i64], ptr %local, align 8
  store volatile [20 x i64] %val2, ptr @global, align 8

; CHECK: add sp, sp, #[[STACKSIZE]]
  ret void
}
