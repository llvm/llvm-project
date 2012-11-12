; RUN: llc < %s -mtriple=i686-apple-darwin8 -mcpu=yonah -march=x86 | FileCheck %s

; These testcases shouldn't require loading into an XMM register then storing 
; to memory, then reloading into an FPStack reg.

; CHECK: test1
; CHECK: fldl
; CHECK-NEXT: ret
define double @test1(double *%P) {
        %A = load double* %P
        ret double %A
}

; fastcc should return a value
; CHECK: test2
; CHECK-NOT: xmm
; CHECK: ret
define fastcc double @test2(<2 x double> %A) {
	%B = extractelement <2 x double> %A, i32 0
	ret double %B
}

; CHECK: test3
; CHECK: sub{{.*}}%esp
; CHECK-NOT: xmm
define fastcc double @test3(<4 x float> %A) {
	%B = bitcast <4 x float> %A to <2 x double>
	%C = call fastcc double @test2(<2 x double> %B)
	ret double %C
}

; Clear the stack when not using a return value.
; CHECK: test4
; CHECK: call
; CHECK: fstp
; CHECK: ret
define void @test4(double *%P) {
  %A = call double @test1(double *%P)
  ret void
}
