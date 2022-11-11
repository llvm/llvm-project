; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)

declare void @throw()
declare i16 @f()

define i16 @test1(i16 %a, ptr %b) personality ptr @__CxxFrameHandler3 {
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
  catchpad within %cs [ptr null, i32 8, ptr null]
  call void @throw() noreturn
  br label %unreachable

catch.2:
  catchpad within %cs [ptr null, i32 64, ptr null]
  store i8 1, ptr %b
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

define i16 @test2(i16 %a, ptr %b) personality ptr @__CxxFrameHandler3 {
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
  catchpad within %cs [ptr null, i32 8, ptr null]
  call void @throw() noreturn
  br label %unreachable

catch.2:
  %c2 = catchpad within %cs [ptr null, i32 32, ptr null]
  store i8 1, ptr %b
  catchret from %c2 to label %cleanup

catch.3:
  %c3 = catchpad within %cs [ptr null, i32 64, ptr null]
  store i8 2, ptr %b
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

declare void @g()

define void @test3() optsize personality ptr @__CxxFrameHandler3 {
entry:
  switch i32 undef, label %if.end57 [
    i32 64, label %sw.bb
    i32 128, label %sw.epilog
    i32 256, label %if.then56
    i32 1024, label %sw.bb
    i32 4096, label %sw.bb33
    i32 16, label %sw.epilog
    i32 8, label %sw.epilog
    i32 32, label %sw.bb44
  ]

sw.bb:
  unreachable

sw.bb33:
  br i1 undef, label %if.end57, label %while.cond.i163.preheader

while.cond.i163.preheader:
  unreachable

sw.bb44:
  %temp0 = load ptr, ptr undef
  invoke void %temp0()
          to label %if.end57 unwind label %catch.dispatch

sw.epilog:
  %temp1 = load ptr, ptr undef
  br label %if.end57

catch.dispatch:
  %cs = catchswitch within none [label %catch1, label %catch2, label %catch3] unwind to caller

catch1:
  %c1 = catchpad within %cs [ptr null, i32 8, ptr null]
  unreachable

catch2:
  %c2 = catchpad within %cs [ptr null, i32 32, ptr null]
  unreachable

catch3:
  %c3 = catchpad within %cs [ptr null, i32 64, ptr null]
  unreachable

if.then56:
  call void @g()
  br label %if.end57

if.end57:
  ret void
}

; This test exercises a complex case that produced an infinite loop during
; compilation when the two cases above did not. The multiple targets from the
; entry switch are not actually fundamental to the failure, but they are
; necessary to suppress various control flow optimizations that would prevent
; the conditions that lead to the failure.
;
; CHECK-LABEL: .def     test3;

