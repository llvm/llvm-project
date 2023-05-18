; Test incoming i128 arguments.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Do some arithmetic so that we can see the register being used.
define void @f1(ptr %r2, i16 %r3, i32 %r4, i64 %r5, i128 %r6) {
; CHECK-LABEL: f1:
; CHECK-DAG:  lg [[REGL:%r[0-5]+]], 8(%r6)
; CHECK-DAG:  lg [[REGH:%r[0-5]+]], 0(%r6)
; CHECK:      algr [[REGL]], [[REGL]]
; CHECK-NEXT: alcgr [[REGH]], [[REGH]]
; CHECK-DAG:  stg [[REGL]], 8(%r2)
; CHECK-DAG:  stg [[REGH]], 0(%r2)
; CHECK:      br %r14
  %y = add i128 %r6, %r6
  store i128 %y, ptr %r2
  ret void
}

; Test a case where the i128 address is passed on the stack.
define void @f2(ptr %r2, i16 %r3, i32 %r4, i64 %r5,
                i128 %r6, i64 %s1, i64 %s2, i128 %s4) {
; CHECK-LABEL: f2:
; CHECK:      lg [[ADDR:%r[1-5]+]], 176(%r15)
; CHECK-DAG:  lg [[REGL:%r[0-5]+]], 8([[ADDR]])
; CHECK-DAG:  lg [[REGH:%r[0-5]+]], 0([[ADDR]])
; CHECK:      algr [[REGL]], [[REGL]]
; CHECK-NEXT: alcgr [[REGH]], [[REGH]]
; CHECK-DAG:  stg [[REGL]], 8(%r2)
; CHECK-DAG:  stg [[REGH]], 0(%r2)
; CHECK:      br %r14
  %y = add i128 %s4, %s4
  store i128 %y, ptr %r2
  ret void
}

; Explicit i128 return values are likewise passed indirectly.
define i128 @f14(i128 %r3) {
; CHECK-LABEL: f14:
; CHECK-DAG:  lg [[REGL:%r[0-5]+]], 8(%r3)
; CHECK-DAG:  lg [[REGH:%r[0-5]+]], 0(%r3)
; CHECK:      algr [[REGL]], [[REGL]]
; CHECK-NEXT: alcgr [[REGH]], [[REGH]]
; CHECK-DAG:  stg [[REGL]], 8(%r2)
; CHECK-DAG:  stg [[REGH]], 0(%r2)
; CHECK:      br %r14
  %y = add i128 %r3, %r3
  ret i128 %y
}

