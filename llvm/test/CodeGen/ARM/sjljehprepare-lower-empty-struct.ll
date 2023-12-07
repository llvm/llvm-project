; RUN: llc -mtriple=armv7-apple-ios -O0 < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios -O1 < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios -O2 < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios -O3 < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-watchos -O3 < %s | FileCheck %s
; RUN: llc -mtriple=armv7k-apple-ios < %s | FileCheck %s --check-prefix=CHECK-WATCH
; RUN: llc -mtriple=armv7-linux -exception-model sjlj -O3 < %s | FileCheck %s --check-prefix=CHECK-LINUX

; SjLjEHPrepare shouldn't crash when lowering empty structs.
;
; Checks that between in case of empty structs used as arguments
; nothing happens, i.e. there are no instructions between
; __Unwind_SjLj_Register and actual @bar invocation


define ptr @foo(i8 %a, {} %c) personality ptr @baz {
entry:
; CHECK: bl __Unwind_SjLj_Register
; CHECK-NEXT: {{[A-Z][a-zA-Z0-9]*}}:
; CHECK-NEXT: bl _bar
; CHECK: bl __Unwind_SjLj_Resume

; CHECK-LINUX: bl _Unwind_SjLj_Register
; CHECK-LINUX-NEXT: .{{[A-Z][a-zA-Z0-9]*}}:
; CHECK-LINUX-NEXT: bl bar
; CHECK-LINUX: bl _Unwind_SjLj_Resume

; CHECK-WATCH-NOT: bl __Unwind_SjLj_Register

  invoke void @bar ()
    to label %unreachable unwind label %handler

unreachable:
  unreachable

handler:
  %tmp = landingpad { ptr, i32 }
  cleanup
  resume { ptr, i32 } undef
}

declare void @bar()
declare i32 @baz(...)
