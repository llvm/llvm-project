; RUN: opt -p sjlj-eh-prepare %s -S -o - | FileCheck %s

; Check that callsites are set up correctly:
; 1. Throwing call in the entry block does not set call_site
;    (function context hasn't been configured yet).
; 2. Throwing call not in the entry block sets call_site to -1
;    (reset to the initial state).
; 3. Invoke instructions set call_site to the correct call site number.
; 4. Resume instruction sets call_site to -1 (reset to the initial state).

define void @test_call_sites() personality ptr @__gxx_personality_sj0 {
entry:
  ; CHECK-NOT: store volatile
  ; CHECK:     call void @may_throw()
  call void @may_throw()

  ; CHECK:      store volatile i32 1
  ; CHECK-NEXT: call void @llvm.eh.sjlj.callsite(i32 1)
  ; CHECK-NEXT: invoke void @may_throw()
  invoke void @may_throw() to label %invoke.cont unwind label %lpad

invoke.cont:
  ; CHECK:      store volatile i32 2
  ; CHECK-NEXT: call void @llvm.eh.sjlj.callsite(i32 2)
  ; CHECK-NEXT: invoke void @may_throw()
  invoke void @may_throw() to label %try.cont unwind label %lpad

lpad:
  ; CHECK:      store volatile i32 -1
  ; CHECK-NEXT: resume
  %lp = landingpad { ptr, i32 } catch ptr @type_info
  resume { ptr, i32 } %lp

try.cont:
  ; CHECK:      store volatile i32 -1
  ; CHECK-NEXT: call void @may_throw
  call void @may_throw()
  ret void
}

@type_info = external constant ptr

declare void @may_throw()
declare i32 @__gxx_personality_sj0(...)
