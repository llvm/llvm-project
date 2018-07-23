; Test strcpy using MVST.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i8 *@strcpy(i8 *%dest, i8 *%src)
declare i8 *@stpcpy(i8 *%dest, i8 *%src)

; Check strcpy.
define i8 *@f1(i8 *%dest, i8 *%src) {
; CHECK-LABEL: f1:
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: lgr [[REG:%r[145]]], %r2
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: mvst [[REG]], %r3
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NOT: %r2
; CHECK: br %r14
  %res = call i8 *@strcpy(i8 *%dest, i8 *%src)
  ret i8 *%res
}

; Check stpcpy.
define i8 *@f2(i8 *%dest, i8 *%src) {
; CHECK-LABEL: f2:
; CHECK: lhi %r0, 0
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: mvst %r2, %r3
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NOT: %r2
; CHECK: br %r14
  %res = call i8 *@stpcpy(i8 *%dest, i8 *%src)
  ret i8 *%res
}

; Check correct operation with other loads and stores.  The load must
; come before the loop and the store afterwards.
define i32 @f3(i32 %dummy, i8 *%dest, i8 *%src, i32 *%resptr, i32 *%storeptr) {
; CHECK-LABEL: f3:
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: l %r2, 0(%r5)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: mvst %r3, %r4
; CHECK-NEXT: jo [[LABEL]]
; CHECK: mvhi 0(%r6), 0
; CHECK: br %r14
  %res = load i32, i32 *%resptr
  %unused = call i8 *@strcpy(i8 *%dest, i8 *%src)
  store i32 0, i32 *%storeptr
  ret i32 %res
}
