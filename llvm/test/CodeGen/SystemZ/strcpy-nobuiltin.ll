; Test that strcmp won't be converted to MVST if calls are
; marked with nobuiltin, eg. for sanitizers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i8 *@strcpy(i8 *%dest, i8 *%src)
declare i8 *@stpcpy(i8 *%dest, i8 *%src)

; Check strcpy.
define i8 *@f1(i8 *%dest, i8 *%src) {
; CHECK-LABEL: f1:
; CHECK-NOT: mvst
; CHECK: brasl %r14, strcpy
; CHECK: br %r14
  %res = call i8 *@strcpy(i8 *%dest, i8 *%src) nobuiltin
  ret i8 *%res
}

; Check stpcpy.
define i8 *@f2(i8 *%dest, i8 *%src) {
; CHECK-LABEL: f2:
; CHECK-NOT: mvst
; CHECK: brasl %r14, stpcpy
; CHECK: br %r14
  %res = call i8 *@stpcpy(i8 *%dest, i8 *%src) nobuiltin
  ret i8 *%res
}

; Check correct operation with other loads and stores.  The load must
; come before the loop and the store afterwards.
define i32 @f3(i32 %dummy, i8 *%dest, i8 *%src, i32 *%resptr, i32 *%storeptr) {
; CHECK-LABEL: f3:
; CHECK-DAG: l [[REG1:%r[0-9]+]], 0(%r5)
; CHECK-NOT: mvst
; CHECK: brasl %r14, strcpy
; CHECK: mvhi 0(%r6), 0
; CHECK: br %r14
  %res = load i32, i32 *%resptr
  %unused = call i8 *@strcpy(i8 *%dest, i8 *%src) nobuiltin
  store i32 0, i32 *%storeptr
  ret i32 %res
}
