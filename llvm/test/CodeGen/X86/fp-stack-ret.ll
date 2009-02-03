; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=i686-apple-darwin8 -mcpu=yonah -march=x86 > %t
; RUN: grep fldl %t | count 1
; RUN: not grep xmm %t
; RUN: grep {sub.*esp} %t | count 1

; These testcases shouldn't require loading into an XMM register then storing 
; to memory, then reloading into an FPStack reg.

define double @test1(double *%P) {
        %A = load double* %P
        ret double %A
}

; fastcc should return a value 
define fastcc double @test2(<2 x double> %A) {
	%B = extractelement <2 x double> %A, i32 0
	ret double %B
}

define fastcc double @test3(<4 x float> %A) {
	%B = bitcast <4 x float> %A to <2 x double>
	%C = call fastcc double @test2(<2 x double> %B)
	ret double %C
}
	
