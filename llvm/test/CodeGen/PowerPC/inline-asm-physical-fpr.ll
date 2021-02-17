; RUN: llc -mcpu=pwr7 -mattr=-altivec -verify-machineinstrs \
; RUN:     -mtriple=powerpc-unknown-aix < %s  | FileCheck %s

; RUN: llc -mcpu=pwr7 -mattr=-altivec -verify-machineinstrs \
; RUN:     -mtriple=powerpc64-unknown-aix < %s | FileCheck %s


define dso_local double @test_double(double %a, double %b) {
entry:
  %0 = tail call double asm "fadd. $0,$1,$2\0A", "={f31},d,d,0"(double %a, double %b, double 0.000000e+00)
  ret double %0
}

; CHECK-LABEL: test_double
; CHECK:         #APP
; CHECK-NEXT:    fadd. 31,1,2

define dso_local signext i32 @test_int(double %a, double %b) {
entry:
  %0 = tail call i32 asm "fadd. $0,$1,$2\0A", "={f0},d,d,0"(double %a, double %b, i32 0)
  ret i32 %0
}

; CHECK-LABEL: test_int
; CHECK:         #APP
; CHECK-NEXT:    fadd. 0,1,2
