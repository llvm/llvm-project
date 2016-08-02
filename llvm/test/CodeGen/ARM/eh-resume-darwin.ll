; RUN: llc < %s -mtriple=armv7-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=IOS
; RUN: llc < %s -mtriple=armv7k-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=IOS
; RUN: llc < %s -mtriple=armv7k-apple-watchos -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=WATCHOS

declare void @func()

declare i32 @__gxx_personality_sj0(...)

define void @test0() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  invoke void @func()
    to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %exn = landingpad { i8*, i32 }
           cleanup
  resume { i8*, i32 } %exn
}

; IOS: __Unwind_SjLj_Resume
; WATCHOS: __Unwind_Resume
