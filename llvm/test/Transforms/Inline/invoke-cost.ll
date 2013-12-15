; RUN: opt -inline < %s -S -o - -inline-threshold=100 | FileCheck %s

target datalayout = "p:32:32"

@glbl = external global i32

declare void @f()
declare i32 @__gxx_personality_v0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @_ZSt9terminatev()

define void @inner1() {
entry:
  invoke void @f() to label %cont1 unwind label %terminate.lpad

cont1:
  invoke void @f() to label %cont2 unwind label %terminate.lpad

cont2:
  invoke void @f() to label %cont3 unwind label %terminate.lpad

cont3:
  invoke void @f() to label %cont4 unwind label %terminate.lpad

cont4:
  ret void

terminate.lpad:
  landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
            catch i8* null
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable
}

define void @outer1() {
; CHECK-LABEL: @outer1(
;
; This call should not get inlined because inner1 actually calls a function
; many times, but it only does so through invoke as opposed to call.
;
; CHECK: call void @inner1
  call void @inner1()
  ret void
}
