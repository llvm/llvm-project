; RUN: llc -enable-new-pm -mtriple=x86_64-unknown -stop-after=x86-isel %s -o - | llc -passes='print<live-vars>' -x mir 2>&1 | FileCheck %s

define i32 @foo(i32 noundef %0) local_unnamed_addr {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %13, label %3

3:                                                ; preds = %1
  %4 = add i32 %0, -1
  %5 = zext i32 %4 to i33
  %6 = add i32 %0, -2
  %7 = zext i32 %6 to i33
  %8 = mul i33 %5, %7
  %9 = lshr i33 %8, 1
  %10 = trunc i33 %9 to i32
  %11 = add i32 %10, %0
  %12 = add i32 %11, -1
  br label %13

13:                                               ; preds = %3, %1
  %14 = phi i32 [ 0, %1 ], [ %12, %3 ]
  ret i32 %14
}

; CHECK: Live variables in machine function: foo
; CHECK: Virtual register '%0':
; CHECK:   Alive in blocks:
; CHECK:   Killed by: No instructions.
