; RUN: llvm-as < %s | opt -constprop | llvm-dis | not grep call

declare double @cos(double)

declare double @sin(double)

declare double @tan(double)

declare double @sqrt(double)

declare i1 @llvm.isunordered.f64(double, double)

define double @T() {
        %A = call double @cos( double 0.000000e+00 )            ; <double> [#uses=1]
        %B = call double @sin( double 0.000000e+00 )            ; <double> [#uses=1]
        %a = add double %A, %B          ; <double> [#uses=1]
        %C = call double @tan( double 0.000000e+00 )            ; <double> [#uses=1]
        %b = add double %a, %C          ; <double> [#uses=1]
        %D = call double @sqrt( double 4.000000e+00 )           ; <double> [#uses=1]
        %c = add double %b, %D          ; <double> [#uses=1]
        ret double %c
}

define i1 @TNAN() {
        %A = fcmp uno double 0x7FF8000000000000, 1.000000e+00           ; <i1> [#uses=1]
        %B = fcmp uno double 1.230000e+02, 1.000000e+00         ; <i1> [#uses=1]
        %C = or i1 %A, %B               ; <i1> [#uses=1]
        ret i1 %C
}

