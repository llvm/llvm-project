; RUN: llc < %s -tailcallopt -march=x86-64 -mattr=+sse2 -mtriple=x86_64-apple-darwin | grep fstpt
; RUN: llc < %s -tailcallopt -march=x86-64 -mattr=+sse2 -mtriple=x86_64-apple-darwin | grep xmm

; Check that x86-64 tail calls support x86_fp80 and v2f32 types. (Tail call
; calling convention out of sync with standard c calling convention on x86_64)
; Bug 4278.

declare fastcc double @tailcallee(x86_fp80, <2 x float>) 
	
define fastcc double @tailcall() {
entry:
  %tmp = fpext float 1.000000e+00 to x86_fp80
	%tmp2 = tail call fastcc double @tailcallee( x86_fp80 %tmp,  <2 x float> <float 1.000000e+00, float 1.000000e+00>)
	ret double %tmp2
}
