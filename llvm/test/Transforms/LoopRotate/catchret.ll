; RUN: opt < %s -loop-rotate -verify-memoryssa -S | FileCheck %s

target triple = "x86_64-pc-windows-msvc"

declare void @always_throws()

define i32 @test() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @always_throws()
          to label %continue unwind label %catch.dispatch

continue:
  unreachable

catch.dispatch:
  %t0 = catchswitch within none [label %catch] unwind to caller

catch:
  %t1 = catchpad within %t0 [i8* null, i32 64, i8* null]
  catchret from %t1 to label %for.cond

for.cond:
  %sum = phi i32 [ %add, %for.body ], [ 0, %catch ]
  %i = phi i32 [ %inc, %for.body ], [ 0, %catch ]
  %cmp = icmp slt i32 %i, 1
  br i1 %cmp, label %for.body, label %return

for.body:
  %add = add nsw i32 1, %sum
  %inc = add nsw i32 %i, 1
  br label %for.cond

return:
  ret i32 0
}

; CHECK: catch:
; CHECK-NEXT: catchpad
; CHECK-NEXT: catchret

declare i32 @__CxxFrameHandler3(...)
