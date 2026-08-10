; Test all condition-code masks that are relevant for CLRJ.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()
@g1 = global i16 0

define void @f1(i32 %target) {
; CHECK-LABEL: f1:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clrjle %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp ule i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(i32 %target) {
; CHECK-LABEL: f2:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clrjl %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp ult i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(i32 %target) {
; CHECK-LABEL: f3:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clrjh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp ugt i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(i32 %target) {
; CHECK-LABEL: f4:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clrjhe %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp uge i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
