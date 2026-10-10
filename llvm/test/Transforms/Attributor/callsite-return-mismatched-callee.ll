; RUN: opt -passes=verify -disable-output %s
; RUN: opt -passes=attributor-cgscc -disable-output %s

define internal void @callee_void() {
  ret void
}

define ptr @test(ptr %fp) {
  %result = call ptr %fp(), !callees !0
  ret ptr %result
}

!0 = !{ptr @callee_void}