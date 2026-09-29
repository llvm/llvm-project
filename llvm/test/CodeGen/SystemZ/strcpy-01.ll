; Test strcpy using MVST.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare ptr@strcpy(ptr %dest, ptr %src)
declare ptr@stpcpy(ptr %dest, ptr %src)

; Check strcpy.
define ptr@f1(ptr %dest, ptr %src) {
; CHECK-LABEL: f1:
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: lgr [[REG:%r[145]]], %r2
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: mvst [[REG]], %r3
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NOT: %r2
; CHECK: br %r14
  %res = call ptr@strcpy(ptr %dest, ptr %src)
  ret ptr %res
}

; Check stpcpy.
define ptr@f2(ptr %dest, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lhi %r0, 0
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: mvst %r2, %r3
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NOT: %r2
; CHECK: br %r14
  %res = call ptr@stpcpy(ptr %dest, ptr %src)
  ret ptr %res
}

; Check correct operation with other loads and stores.  The load must
; come before the loop and the store afterwards.
define i32 @f3(i32 %dummy, ptr %dest, ptr %src, ptr %resptr, ptr %storeptr) {
; CHECK-LABEL: f3:
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: l %r2, 0(%r5)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: mvst %r3, %r4
; CHECK-NEXT: jo [[LABEL]]
; CHECK: mvhi 0(%r6), 0
; CHECK: br %r14
  %res = load i32, ptr %resptr
  %unused = call ptr@strcpy(ptr %dest, ptr %src)
  store i32 0, ptr %storeptr
  ret i32 %res
}
