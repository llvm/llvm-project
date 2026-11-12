; DISABLED: llc < %s -mtriple=mips -o %t
; DISABLED: grep {lui.*32767} %t | count 1
; DISABLED: grep {ori.*65535} %t | count 1
; RUN: false
; XFAIL: *

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-unknown-psp-elf"

define double @A(double %c, double %d) nounwind readnone  {
entry:
	tail call double @fabs( double %c ) nounwind readnone 		; <double>:0 [#uses=1]
	tail call double @fabs( double %d ) nounwind readnone 		; <double>:0 [#uses=1]
  fadd double %0, %1
  ret double %2
}

declare double @fabs(double) nounwind readnone 
