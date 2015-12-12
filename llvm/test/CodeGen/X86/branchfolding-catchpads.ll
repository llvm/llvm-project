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
  %cs = catchswitch within none [ label %catch, label %catch.2 ] unwind to caller

catch:
  catchpad within %cs [i8* null, i32 8, i8* null]
  call void @throw() noreturn
  br label %unreachable

catch.2:
  catchpad within %cs [i8* null, i32 64, i8* null]
  store i8 1, i8* %b
  call void @throw() noreturn
  br label %unreachable

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
  %cs = catchswitch within none [ label %catch, label %catch.2, label %catch.3 ] unwind to caller

catch:
  catchpad within %cs [i8* null, i32 8, i8* null]
  call void @throw() noreturn
  br label %unreachable

catch.2:
  %c2 = catchpad within %cs [i8* null, i32 32, i8* null]
  store i8 1, i8* %b
  catchret from %c2 to label %cleanup

catch.3:
  %c3 = catchpad within %cs [i8* null, i32 64, i8* null]
  store i8 2, i8* %b
  catchret from %c3 to label %cleanup

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

