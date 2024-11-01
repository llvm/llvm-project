; RUN: opt -passes=indvars -S < %s | FileCheck %s

declare void @use(i32)
declare void @use.i8(i8)

define void @f() {
; CHECK-LABEL: @f
 entry:
  br label %loop

 loop:
; The only use for idx.mirror is to induce an nuw for %idx.  It does
; not induce an nuw for %idx.inc
  %idx.mirror = phi i8 [ -6, %entry ], [ %idx.mirror.inc, %loop ]
  %idx = phi i8 [ -5, %entry ], [ %idx.inc, %loop ]

  %idx.sext = sext i8 %idx to i32
  call void @use(i32 %idx.sext)

  %idx.mirror.inc = add nuw i8 %idx.mirror, 1
  call void @use.i8(i8 %idx.mirror.inc)

  %idx.inc = add i8 %idx, 1
; CHECK-NOT: %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %cmp = icmp ugt i8 %idx.inc, 0
  br i1 %cmp, label %loop, label %exit

 exit:
  ret void
}
