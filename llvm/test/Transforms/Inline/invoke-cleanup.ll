; RUN: opt %s -inline -S | FileCheck %s
; RUN: opt %s -passes='cgscc(inline)' -S | FileCheck %s

declare void @external_func()

@exception_type1 = external global i8
@exception_type2 = external global i8


define internal void @inner() personality i8* null {
  invoke void @external_func()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32
      catch i8* @exception_type1
  resume i32 %lp
}

; Test that the "cleanup" clause is kept when inlining @inner() into
; this call site (PR17872), otherwise C++ destructors will not be
; called when they should be.

define void @outer() personality i8* null {
  invoke void @inner()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32
      cleanup
      catch i8* @exception_type2
  resume i32 %lp
}
; CHECK: define void @outer
; CHECK: landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: catch i8* @exception_type1
; CHECK-NEXT: catch i8* @exception_type2
