; RUN: llc -mtriple=x86_64 -O0 < %s | FileCheck %s

; Check that we don't crash on this input.
; CHECK-LABEL: @foo
; CHECK: __stack_chk_guard
; CHECK: retq
define hidden void @foo(ptr %ptr) #0 {
entry:
  %args.addr = alloca ptr, align 8
  %0 = va_arg ptr %args.addr, ptr
  store ptr %0, ptr %ptr
  ret void
}

attributes #0 = { sspstrong }
attributes #1 = { optsize }

