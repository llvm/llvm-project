; RUN: llc < %s -mtriple=armv7-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=IOS
; RUN: llc < %s -mtriple=armv7-apple-watchos -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=IOS
; RUN: llc < %s -mtriple=armv7k-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=WATCHABI
; RUN: llc < %s -mtriple=armv7k-apple-watchos -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=WATCHABI
; RUN: llc < %s -mtriple=armv7-none-gnueabihf -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=EABI
; RUN: llc < %s -mtriple=armv7-none-none -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=ABI

; ARM EABI for C++/__gxx_personality* specifies __cxa_end_cleanup, but for C code / __gcc_personality*
; the _Unwind_Resume is required.

declare void @func()

declare i32 @__gcc_personality_v0(...)

define void @test0() personality ptr @__gcc_personality_v0 {
entry:
  invoke void @func()
    to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %exn = landingpad { ptr, i32 }
           cleanup
  resume { ptr, i32 } %exn
}

; IOS: __Unwind_SjLj_Resume
; WATCHABI: __Unwind_Resume
; EABI: _Unwind_Resume
; ABI: _Unwind_Resume
