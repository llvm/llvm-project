; RUN: opt -passes=inline -inline-threshold=10 -S < %s | FileCheck %s

declare void @foo()

; CHECK-LABEL: @caller
; CHECK-NOT:   %res = call i64 @callee(ptr %p)
define i64 @caller(ptr %p) {
  %res = call i64 @callee(ptr %p)
  ret i64 %res
}

define i64 @callee(ptr %p) {
  %null_check = icmp eq ptr %p, null
  br i1 %null_check, label %is_null, label %non_null, !make.implicit !0

is_null:
  call void @foo()
  ret i64 0

non_null:
  ret i64 1
}

!0 = !{}
