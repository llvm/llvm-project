; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)

declare void @throw()
declare i16 @f()

define i16 @test1(i16 %a, i8* %b) personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %cmp = icmp eq i16 %a, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %call1 = invoke i16 @f()
          to label %cleanup unwind label %catch.dispatch

if.else:
  %call2 = invoke i16 @f()
          to label %cleanup unwind label %catch.dispatch

catch.dispatch:
  catchpad [i8* null, i32 8, i8* null]
          to label %catch unwind label %catch.dispatch.2

catch:
  invoke void @throw() noreturn
          to label %unreachable unwind label %catchendblock

catch.dispatch.2:
  catchpad [i8* null, i32 64, i8* null]
          to label %catch.2 unwind label %catchendblock

catch.2:
  store i8 1, i8* %b
  invoke void @throw() noreturn
          to label %unreachable unwind label %catchendblock

catchendblock:
  catchendpad unwind to caller

cleanup:
  %retval = phi i16 [ %call1, %if.then ], [ %call2, %if.else ]
  ret i16 %retval

unreachable:
  unreachable
}

; This test verifies the case where two funclet blocks meet the old criteria
; to be placed at the end.  The order of the blocks is not important for the
; purposes of this test.  The failure mode is an infinite loop during
; compilation.
;
; CHECK-LABEL: .def     test1;

define i16 @test2(i16 %a, i8* %b) personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %cmp = icmp eq i16 %a, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %call1 = invoke i16 @f()
          to label %cleanup unwind label %catch.dispatch

if.else:
  %call2 = invoke i16 @f()
          to label %cleanup unwind label %catch.dispatch

catch.dispatch:
  catchpad [i8* null, i32 8, i8* null]
          to label %catch unwind label %catch.dispatch.2

catch:
  invoke void @throw() noreturn
          to label %unreachable unwind label %catchendblock

catch.dispatch.2:
  %c2 = catchpad [i8* null, i32 32, i8* null]
          to label %catch.2 unwind label %catch.dispatch.3

catch.2:
  store i8 1, i8* %b
  catchret %c2 to label %cleanup

catch.dispatch.3:
  %c3 = catchpad [i8* null, i32 64, i8* null]
          to label %catch.3 unwind label %catchendblock

catch.3:
  store i8 2, i8* %b
  catchret %c3 to label %cleanup

catchendblock:
  catchendpad unwind to caller

cleanup:
  %retval = phi i16 [ %call1, %if.then ], [ %call2, %if.else ], [ -1, %catch.2 ], [ -1, %catch.3 ]
  ret i16 %retval

unreachable:
  unreachable
}

; This test verifies the case where three funclet blocks all meet the old
; criteria to be placed at the end.  The order of the blocks is not important
; for the purposes of this test.  The failure mode is an infinite loop during
; compilation.
;
; CHECK-LABEL: .def     test2;

