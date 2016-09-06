; Test variable-sized allocas and addresses based on them in cases where
; stack arguments are needed.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-A
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-B
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-C
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-D
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-FP

declare i64 @bar(i8 *%a, i8 *%b, i8 *%c, i8 *%d, i8 *%e, i64 %f, i64 %g)

; Allocate %length bytes and take addresses based on the result.
; There are two stack arguments, so an offset of 160 + 2 * 8 == 176
; is added to the copy of %r15.
define i64 @f1(i64 %length, i64 %index) {
; FIXME: a better sequence would be:
;
;    lgr %r1, %r15
;    sgr %r1, %r2
;    nill %r1, 0xfff8
;    lgr %r15, %r1
;
; CHECK-LABEL: f1:
; CHECK-DAG: la [[REG1:%r[0-5]]], 7(%r2)
; CHECK-DAG: nill [[REG1]], 65528
; CHECK-DAG: lgr [[REG2:%r[0-5]]], %r15
; CHECK: sgr [[REG2]], [[REG1]]
; CHECK: lgr %r15, [[REG2]]
;
; CHECK-A-LABEL: f1:
; CHECK-A: lgr %r15, %r1
; CHECK-A: la %r2, 176(%r1)
;
; CHECK-B-LABEL: f1:
; CHECK-B: lgr %r15, %r1
; CHECK-B: la %r3, 177(%r1)
;
; CHECK-C-LABEL: f1:
; CHECK-C: lgr %r15, %r1
; CHECK-C: la %r4, 4095({{%r3,%r1|%r1,%r3}})
;
; CHECK-D-LABEL: f1:
; CHECK-D: lgr %r15, %r1
; CHECK-D: lay %r5, 4096({{%r3,%r1|%r1,%r3}})
;
; CHECK-E-LABEL: f1:
; CHECK-E: lgr %r15, %r1
; CHECK-E: lay %r6, 4271({{%r3,%r1|%r1,%r3}})
;
; CHECK-FP-LABEL: f1:
; CHECK-FP: lgr %r11, %r15
; CHECK-FP: lmg %r6, %r15, 224(%r11)
  %a = alloca i8, i64 %length
  %b = getelementptr i8, i8 *%a, i64 1
  %cindex = add i64 %index, 3919
  %c = getelementptr i8, i8 *%a, i64 %cindex
  %dindex = add i64 %index, 3920
  %d = getelementptr i8, i8 *%a, i64 %dindex
  %eindex = add i64 %index, 4095
  %e = getelementptr i8, i8 *%a, i64 %eindex
  %count = call i64 @bar(i8 *%a, i8 *%b, i8 *%c, i8 *%d, i8 *%e, i64 0, i64 0)
  %res = add i64 %count, 1
  ret i64 %res
}
