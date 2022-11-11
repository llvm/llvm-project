; Test that strcmp won't be converted to CLST if calls are
; marked with nobuiltin, eg. for sanitizers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare signext i32 @strcmp(ptr %src1, ptr %src2)

; Check a case where the result is used as an integer.
define i32 @f1(ptr %src1, ptr %src2) {
; CHECK-LABEL: f1:
; CHECK-NOT: clst
; CHECK: brasl %r14, strcmp
; CHECK: br %r14
  %res = call i32 @strcmp(ptr %src1, ptr %src2) nobuiltin
  ret i32 %res
}

; Check a case where the result is tested for equality.
define void @f2(ptr %src1, ptr %src2, ptr %dest) {
; CHECK-LABEL: f2:
; CHECK-NOT: clst
; CHECK: brasl %r14, strcmp
; CHECK: br %r14
  %res = call i32 @strcmp(ptr %src1, ptr %src2) nobuiltin
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, ptr %dest
  br label %exit

exit:
  ret void
}

; Test a case where the result is used both as an integer and for
; branching.
define i32 @f3(ptr %src1, ptr %src2, ptr %dest) {
; CHECK-LABEL: f3:
; CHECK-NOT: clst
; CHECK: brasl %r14, strcmp
; CHECK: br %r14
entry:
  %res = call i32 @strcmp(ptr %src1, ptr %src2) nobuiltin
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, ptr %dest
  br label %exit

exit:
  ret i32 %res
}
