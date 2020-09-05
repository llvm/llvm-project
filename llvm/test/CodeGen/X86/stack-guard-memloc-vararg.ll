; RUN: llc -mtriple=x86_64 -O0 < %s | FileCheck %s

; Check that we don't crash on this input.
; CHECK-LABEL: @foo
; CHECK: __stack_chk_guard
; CHECK: retq
define hidden void @foo(i8** %ptr) #0 {
entry:
  %args.addr = alloca i8*, align 8
  %0 = va_arg i8** %args.addr, i8*
  store i8* %0, i8** %ptr
  ret void
}

attributes #0 = { sspstrong }
attributes #1 = { optsize }

