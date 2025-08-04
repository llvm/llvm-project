;; __stack_chk_fail should have the noreturn attr even if it is an alias
; REQUIRES: x86-registered-target
; RUN: opt -mtriple=x86_64-pc-linux-gnu %s -passes=stack-protector -S | FileCheck %s

define hidden void @__stack_chk_fail_impl() {
  unreachable
}

@__stack_chk_fail = hidden alias void (), ptr @__stack_chk_fail_impl

; CHECK-LABEL: @store_captures(
; CHECK:       CallStackCheckFailBlk:
; CHECK-NEXT:      call void @__stack_chk_fail() [[ATTRS:#.*]]
define void @store_captures() sspstrong {
entry:
  %a = alloca i32, align 4
  %j = alloca ptr, align 8
  store ptr %a, ptr %j, align 8
  ret void
}

; CHECK: attributes [[ATTRS]] = { noreturn }
