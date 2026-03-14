; Test calling functions with multiple return values (LLVM ABI extension)
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Up to four integer return values fit into GPRs.
declare { i64, i64, i64, i64 } @bar1()

define i64 @f1() {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, bar1
; CHECK: lgr %r2, %r5
; CHECK: br %r14
  %mret = call { i64, i64, i64, i64 } @bar1()
  %ret = extractvalue { i64, i64, i64, i64 } %mret, 3
  ret i64 %ret
}

; More than four integer return values use sret.
declare { i64, i64, i64, i64, i64 } @bar2()

define i64 @f2() {
; CHECK-LABEL: f2:
; CHECK: la %r2, 160(%r15)
; CHECK: brasl %r14, bar2
; CHECK: lg  %r2, 192(%r15)
; CHECK: br %r14
  %mret = call { i64, i64, i64, i64, i64 } @bar2()
  %ret = extractvalue { i64, i64, i64, i64, i64 } %mret, 4
  ret i64 %ret
}

; Up to four floating-point return values fit into GPRs.
declare { double, double, double, double } @bar3()

define double @f3() {
; CHECK-LABEL: f3:
; CHECK: brasl %r14, bar3
; CHECK: ldr %f0, %f6
; CHECK: br %r14
  %mret = call { double, double, double, double } @bar3()
  %ret = extractvalue { double, double, double, double } %mret, 3
  ret double %ret
}

; More than four integer return values use sret.
declare { double, double, double, double, double } @bar4()

define double @f4() {
; CHECK-LABEL: f4:
; CHECK: la %r2, 160(%r15)
; CHECK: brasl %r14, bar4
; CHECK: ld  %f0, 192(%r15)
; CHECK: br %r14
  %mret = call { double, double, double, double, double } @bar4()
  %ret = extractvalue { double, double, double, double, double } %mret, 4
  ret double %ret
}
