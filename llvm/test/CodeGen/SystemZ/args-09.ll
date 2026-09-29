; Test the handling of i128 argument values
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-INT
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-I128-1
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-I128-2
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-STACK

declare void @bar(i64, i64, i64, i64, i128,
                  i64, i64, i64, i64, i128)

; There are two indirect i128 slots, one at offset 200 (the first available
; byte after the outgoing arguments) and one immediately after it at 216.
; These slots should be set up outside the glued call sequence, so would
; normally use %f0/%f2 as the first available 128-bit pair.  This choice
; is hard-coded in the I128 tests.
;
; The order of the CHECK-STACK stores doesn't matter.  It would be OK to reorder
; them in response to future code changes.
define void @foo() {
; CHECK-INT-LABEL: foo:
; CHECK-INT-DAG: lghi %r2, 1
; CHECK-INT-DAG: lghi %r3, 2
; CHECK-INT-DAG: lghi %r4, 3
; CHECK-INT-DAG: lghi %r5, 4
; CHECK-INT-DAG: la %r6, {{200|216}}(%r15)
; CHECK-INT: brasl %r14, bar@PLT
;
; CHECK-I128-1-LABEL: foo:
; CHECK-I128-1: aghi %r15, -232
; CHECK-I128-1-DAG: mvghi 200(%r15), 0
; CHECK-I128-1-DAG: mvghi 208(%r15), 0
; CHECK-I128-1: brasl %r14, bar@PLT
;
; CHECK-I128-2-LABEL: foo:
; CHECK-I128-2: aghi %r15, -232
; CHECK-I128-2-DAG: mvghi 216(%r15), 0
; CHECK-I128-2-DAG: mvghi 224(%r15), 0
; CHECK-I128-2: brasl %r14, bar@PLT
;
; CHECK-STACK-LABEL: foo:
; CHECK-STACK: aghi %r15, -232
; CHECK-STACK: la [[REGISTER:%r[0-5]+]], {{200|216}}(%r15)
; CHECK-STACK: stg [[REGISTER]], 192(%r15)
; CHECK-STACK: mvghi 184(%r15), 8
; CHECK-STACK: mvghi 176(%r15), 7
; CHECK-STACK: mvghi 168(%r15), 6
; CHECK-STACK: mvghi 160(%r15), 5
; CHECK-STACK: brasl %r14, bar@PLT

  call void @bar (i64 1, i64 2, i64 3, i64 4, i128 0,
                  i64 5, i64 6, i64 7, i64 8, i128 0)
  ret void
}
