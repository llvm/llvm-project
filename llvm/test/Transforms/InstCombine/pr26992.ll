; RUN: opt -instcombine -S < %s | FileCheck %s
target triple = "x86_64-pc-windows-msvc"

define i1 @test1(i8* %p) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %a = getelementptr i8, i8* %p, i64 1
  invoke void @may_throw()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:
  %b = getelementptr inbounds i8, i8* %a, i64 1
  invoke void @may_throw()
          to label %exit unwind label %catch.dispatch

catch.dispatch:
  %c = phi i8* [ %b, %invoke.cont ], [ %a, %entry ]
  %tmp1 = catchswitch within none [label %catch] unwind to caller

catch:
  %tmp2 = catchpad within %tmp1 [i8* null, i32 64, i8* null]
  catchret from %tmp2 to label %exit

exit:
  %d = phi i8* [ %a, %invoke.cont ], [ %c, %catch ]
  %cmp = icmp eq i8* %d, %a
  ret i1 %cmp
}

; CHECK-LABEL: define i1 @test1(
; CHECK:  %[[gep_a:.*]] = getelementptr i8, i8* %p, i64 1
; CHECK:  %[[gep_b:.*]] = getelementptr inbounds i8, i8* %p, i64 2
; CHECK:  phi i8* [ %[[gep_b]], {{.*}} ], [ %[[gep_a]], {{.*}} ]
; CHECK:  %tmp1 = catchswitch within none [label %catch] unwind to caller

declare void @may_throw()

declare i32 @__CxxFrameHandler3(...)
