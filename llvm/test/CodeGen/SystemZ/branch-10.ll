; Test all condition-code masks that are relevant for CLGRJ.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()
@g1 = global i16 0

define void @f1(i64 %target) {
; CHECK-LABEL: f1:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clgrjle %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp ule i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(i64 %target) {
; CHECK-LABEL: f2:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clgrjl %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp ult i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(i64 %target) {
; CHECK-LABEL: f3:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clgrjh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp ugt i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(i64 %target) {
; CHECK-LABEL: f4:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clgrjhe %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp uge i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
