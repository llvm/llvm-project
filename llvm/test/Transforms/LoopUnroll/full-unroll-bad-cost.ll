; RUN: opt -S -loop-unroll < %s | FileCheck %s
; RUN: opt < %s -passes='require<opt-remark-emit>,loop(unroll-full)' -S | FileCheck %s

; LLVM should not try to fully unroll this loop.

declare void @f()
declare void @g()
declare void @h()

define void @trivial_loop() {
; CHECK-LABEL: @trivial_loop(
 entry:
  br label %loop

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add i32 %idx, 1
  call void @f()
  call void @g()
  call void @h()
  call void @f()
  call void @g()
  call void @h()
  call void @f()
  call void @g()
  call void @h()
  call void @f()
  call void @g()
  call void @h()
  call void @f()
  call void @g()
  call void @h()
  %be = icmp slt i32 %idx, 268435456
  br i1 %be, label %loop, label %exit

; CHECK: loop:
; CHECK-NEXT:  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
; CHECK-NEXT:  %idx.inc = add i32 %idx, 1
; CHECK-NEXT:  call void @f()
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  call void @h()
; CHECK-NEXT:  call void @f()
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  call void @h()
; CHECK-NEXT:  call void @f()
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  call void @h()
; CHECK-NEXT:  call void @f()
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  call void @h()
; CHECK-NEXT:  call void @f()
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  call void @h()
; CHECK-NEXT:  %be = icmp slt i32 %idx, 268435456
; CHECK-NEXT:  br i1 %be, label %loop, label %exit

 exit:
  ret void
}
