; RUN: opt < %s -passes=pseudo-probe -S | FileCheck %s

; CFG hashes in !llvm.pseudo_probe_desc must match between a nounwind/call
; shape and the equivalent invoke shape, so AutoFDO can use one profile.

declare void @f0()
declare void @f1()
declare void @f2()
declare i32 @__gxx_personality_v0(...)

; for (;;) f1();
define void @selfloop_calls() {
entry:
  br label %loop

loop:
  call void @f1()
  br label %loop
}

define void @selfloop_invokes() personality ptr @__gxx_personality_v0 {
entry:
  br label %loop

loop:
  invoke void @f1()
          to label %loop unwind label %lpad

lpad:
  %eh = landingpad { ptr, i32 }
          cleanup
  ret void
}

; f0(); for (;;) f1();  -- loop header is also the invoke continuation
define void @prefix_then_loop_calls() {
entry:
  call void @f0()
  br label %loop

loop:
  call void @f1()
  br label %loop
}

define void @prefix_then_loop_invokes() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @f0()
          to label %loop unwind label %lpad

loop:
  invoke void @f1()
          to label %loop unwind label %lpad

lpad:
  %eh = landingpad { ptr, i32 }
          cleanup
  ret void
}

; for (;;) { f1(); f2(); }
define void @twocall_loop_calls() {
entry:
  br label %loop

loop:
  call void @f1()
  call void @f2()
  br label %loop
}

define void @twocall_loop_invokes() personality ptr @__gxx_personality_v0 {
entry:
  br label %for.cond

for.cond:
  invoke void @f1()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @f2()
          to label %for.cond unwind label %lpad

lpad:
  %eh = landingpad { ptr, i32 }
          cleanup
  ret void
}

; CHECK: !{i64 {{-?[0-9]+}}, i64 [[H1:[0-9]+]], !"selfloop_calls"}
; CHECK: !{i64 {{-?[0-9]+}}, i64 [[H1]], !"selfloop_invokes"}

; CHECK: !{i64 {{-?[0-9]+}}, i64 [[H2:[0-9]+]], !"prefix_then_loop_calls"}
; CHECK: !{i64 {{-?[0-9]+}}, i64 [[H2]], !"prefix_then_loop_invokes"}

; FIXME: The invoke form needs two blocks where the call form needs one, so
; the hashes cannot match yet. Hardcoded so a future fix fails this test.
; CHECK: !{i64 {{-?[0-9]+}}, i64 562986241617760, !"twocall_loop_calls"}
; CHECK: !{i64 {{-?[0-9]+}}, i64 563004266919717, !"twocall_loop_invokes"}
