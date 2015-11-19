; RUN: llc -mtriple=armv7-apple-ios -O0 < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios -O1 < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios -O2 < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios -O3 < %s | FileCheck %s
; RUN: llc -mtriple=armv7k-apple-ios < %s | FileCheck %s

; SjLjEHPrepare shouldn't crash when lowering empty structs.
;
; Checks that between in case of empty structs used as arguments
; nothing happens, i.e. there are no instructions between
; __Unwind_SjLj_Register and actual @bar invocation


define i8* @foo(i8 %a, {} %c) personality i8* bitcast (i32 (...)* @baz to i8*) {
entry:
; CHECK: bl __Unwind_SjLj_Register
; CHECK-NEXT: {{[A-Z][a-zA-Z0-9]*}}:
; CHECK-NEXT: bl _bar
  invoke void @bar ()
    to label %unreachable unwind label %handler

unreachable:
  unreachable

handler:
  %tmp = landingpad { i8*, i32 }
  cleanup
  resume { i8*, i32 } undef
}

declare void @bar()
declare i32 @baz(...)
