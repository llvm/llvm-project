; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s
;
; Test that a CC result of a sub that can overflow is tested with the right predicate.

define i32 @fun0(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: fun0
; CHECK: s %r2, 0(%r4)
; CHECK: bner %r14
entry:
  %cur = load i32, ptr %dest
  %res = sub nsw i32 %a, %cur
  %cmp = icmp ne i32 %a, %cur
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

define i32 @fun1(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: fun1
; CHECK: s %r2, 0(%r4)
; CHECK: bner %r14
entry:
  %cur = load i32, ptr %dest
  %res = sub nuw i32 %a, %cur
  %cmp = icmp ne i32 %a, %cur
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}
