; RUN: llc < %s -mtriple=sparc
; PR1540

declare float @sinf(float)
declare double @sin(double)
define double @test_sin(float %F) {
        %G = call float @sinf( float %F )               ; <float> [#uses=1]
        %H = fpext float %G to double           ; <double> [#uses=1]
        %I = call double @sin( double %H )              ; <double> [#uses=1]
        ret double %I
}
