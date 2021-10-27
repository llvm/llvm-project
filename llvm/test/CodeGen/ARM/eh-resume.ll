; RUN: llc < %s -mtriple=armv7-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=IOS
; RUN: llc < %s -mtriple=armv7-apple-watchos -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=IOS
; RUN: llc < %s -mtriple=armv7k-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=WATCHABI
; RUN: llc < %s -mtriple=armv7k-apple-watchos -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=WATCHABI
; RUN: llc < %s -mtriple=armv7-none-gnueabihf -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=EABI
; RUN: llc < %s -mtriple=armv7-none-none -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=ABI

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
; WATCHABI: __Unwind_Resume
; EABI: __cxa_end_cleanup
; ABI: _Unwind_Resume
