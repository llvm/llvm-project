; RUN: opt -S -passes=indvars < %s | FileCheck %s

declare void @use(i1)

define void @sext_condition(i8 %t) {
; CHECK-LABEL: sext_condition
 entry:
  %st = sext i8 %t to i16
  %ecmp = icmp slt i16 %st, 42
  br i1 %ecmp, label %loop, label %exit

 loop:
; CHECK-LABEL: loop
  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add i8 %idx, 1
  %c = icmp slt i8 %idx, 42
; CHECK: call void @use(i1 true)
  call void @use(i1 %c)
  %be = icmp slt i8 %idx.inc, 42
  br i1 %be, label %loop, label %exit

 exit:
  ret void
}

define void @zext_condition(i8 %t) {
; CHECK-LABEL: zext_condition
 entry:
  %st = zext i8 %t to i16
  %ecmp = icmp ult i16 %st, 42
  br i1 %ecmp, label %loop, label %exit

 loop:
; CHECK-LABEL: loop
  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add i8 %idx, 1
  %c = icmp ult i8 %idx, 42
; CHECK: call void @use(i1 true)
  call void @use(i1 %c)
  %be = icmp ult i8 %idx.inc, 42
  br i1 %be, label %loop, label %exit

 exit:
  ret void
}
