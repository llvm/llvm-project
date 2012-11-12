; RUN: llc < %s -march=arm | FileCheck %s
target triple = "armv6-apple-macosx10.6"

declare void @func()

declare i32 @__gxx_personality_sj0(...)

define void @test0() {
entry:
  invoke void @func()
    to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %exn = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)
           cleanup
  resume { i8*, i32 } %exn
}

; CHECK: __Unwind_SjLj_Resume
