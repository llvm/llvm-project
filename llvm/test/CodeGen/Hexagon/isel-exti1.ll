; RUN: llc -O0 -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: sexti1
; CHECK: r[[REG:[0-9]+]] = mux(p{{[0-3]}},#-1,#0)
; CHECK: combine(r[[REG]],r[[REG]])
define i64 @sexti1(i64 %a0, i64 %a1) {
entry:
  %t0 = icmp ult i64 %a0, %a1
  %t1 = sext i1 %t0 to i64
  ret i64 %t1
}

; CHECK-LABEL: zexti1
; CHECK: r[[REG:[0-9]+]] = mux(p{{[0-3]}},#1,#0)
; CHECK: combine(#0,r[[REG]])
define i64 @zexti1(i64 %a0, i64 %a1) {
entry:
  %t0 = icmp ult i64 %a0, %a1
  %t1 = zext i1 %t0 to i64
  ret i64 %t1
}

