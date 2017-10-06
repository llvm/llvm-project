; Make sure that the alloca offset isn't lost when the alloca result is
; used directly in a load or store.  There must always be an LA or LAY.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-A
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-B
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-C
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-D

declare i64 @bar(i8 *%a)

define i64 @f1(i64 %length, i64 %index) {
; CHECK-A-LABEL: f1:
; CHECK-A-DAG: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-A-DAG: la %r2, 160([[ADDR]])
; CHECK-A: mvi 0(%r2), 0
;
; CHECK-B-LABEL: f1:
; CHECK-B-DAG: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-B-DAG: la %r2, 160([[ADDR]])
; CHECK-B: mvi 4095(%r2), 1
;
; CHECK-C-LABEL: f1:
; CHECK-C-DAG: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-C-DAG: la %r2, 160([[ADDR]])
; CHECK-C-DAG: lhi [[TMP:%r[0-5]]], 2
; CHECK-C: stc [[TMP]], 0({{%r3,%r2|%r2,%r3}})
;
; CHECK-D-LABEL: f1:
; CHECK-D-DAG: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-D-DAG: la %r2, 160([[ADDR]])
; CHECK-D-DAG: lhi [[TMP:%r[0-5]]], 3
; CHECK-D: stc [[TMP]], 4095({{%r3,%r2|%r2,%r3}})
;
; CHECK-E-LABEL: f1:
; CHECK-E-DAG: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-E-DAG: la %r2, 160([[ADDR]])
; CHECK-E-DAG: lhi [[TMP:%r[0-5]]], 4
; CHECK-E: stcy [[TMP]], 4096({{%r3,%r2|%r2,%r3}})
  %a = alloca i8, i64 %length
  store volatile i8 0, i8 *%a
  %b = getelementptr i8, i8 *%a, i64 4095
  store volatile i8 1, i8 *%b
  %c = getelementptr i8, i8 *%a, i64 %index
  store volatile i8 2, i8 *%c
  %d = getelementptr i8, i8 *%c, i64 4095
  store volatile i8 3, i8 *%d
  %e = getelementptr i8, i8 *%d, i64 1
  store volatile i8 4, i8 *%e
  %count = call i64 @bar(i8 *%a)
  %res = add i64 %count, 1
  ret i64 %res
}
