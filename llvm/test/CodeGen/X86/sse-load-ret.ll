; RUN: llc < %s -march=x86 -mcpu=yonah | not grep movss
; RUN: llc < %s -march=x86 -mcpu=yonah | not grep xmm

define double @test1(double* %P) {
        %X = load double, double* %P            ; <double> [#uses=1]
        ret double %X
}

define double @test2() {
        ret double 1.234560e+03
}


; FIXME: Todo
;double %test3(bool %B) {
;	%C = select bool %B, double 123.412, double 523.01123123
;	ret double %C
;}

