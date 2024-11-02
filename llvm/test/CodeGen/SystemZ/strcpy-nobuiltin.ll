; Test that strcmp won't be converted to MVST if calls are
; marked with nobuiltin, eg. for sanitizers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare ptr@strcpy(ptr %dest, ptr %src)
declare ptr@stpcpy(ptr %dest, ptr %src)

; Check strcpy.
define ptr@f1(ptr %dest, ptr %src) {
; CHECK-LABEL: f1:
; CHECK-NOT: mvst
; CHECK: brasl %r14, strcpy
; CHECK: br %r14
  %res = call ptr@strcpy(ptr %dest, ptr %src) nobuiltin
  ret ptr %res
}

; Check stpcpy.
define ptr@f2(ptr %dest, ptr %src) {
; CHECK-LABEL: f2:
; CHECK-NOT: mvst
; CHECK: brasl %r14, stpcpy
; CHECK: br %r14
  %res = call ptr@stpcpy(ptr %dest, ptr %src) nobuiltin
  ret ptr %res
}

; Check correct operation with other loads and stores.  The load must
; come before the loop and the store afterwards.
define i32 @f3(i32 %dummy, ptr %dest, ptr %src, ptr %resptr, ptr %storeptr) {
; CHECK-LABEL: f3:
; CHECK-DAG: l [[REG1:%r[0-9]+]], 0(%r5)
; CHECK-NOT: mvst
; CHECK: brasl %r14, strcpy
; CHECK: mvhi 0(%r6), 0
; CHECK: br %r14
  %res = load i32, ptr %resptr
  %unused = call ptr@strcpy(ptr %dest, ptr %src) nobuiltin
  store i32 0, ptr %storeptr
  ret i32 %res
}
