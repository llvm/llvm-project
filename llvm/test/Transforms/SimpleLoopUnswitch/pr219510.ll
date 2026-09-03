; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial>)' -S < %s | FileCheck %s 

declare void @barrier() memory(read)
declare void @clobber()

; CHECK: call void @barrier()
; CHECK-NEXT: %lv = load i32, ptr %ptr


define i32 @bad_unswitch(ptr %ptr, i32 %N) {
entry:
  br label %loop.header
loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  call void @barrier()
  %lv = load i32, ptr %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}
