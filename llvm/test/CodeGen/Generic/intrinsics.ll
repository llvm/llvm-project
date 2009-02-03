; RUN: llvm-as < %s | llc

;; SQRT
declare float @llvm.sqrt.f32(float)

declare double @llvm.sqrt.f64(double)

define double @test_sqrt(float %F) {
        %G = call float @llvm.sqrt.f32( float %F )              ; <float> [#uses=1]
        %H = fpext float %G to double           ; <double> [#uses=1]
        %I = call double @llvm.sqrt.f64( double %H )            ; <double> [#uses=1]
        ret double %I
}


; SIN
declare float @sinf(float)

declare double @sin(double)

define double @test_sin(float %F) {
        %G = call float @sinf( float %F )               ; <float> [#uses=1]
        %H = fpext float %G to double           ; <double> [#uses=1]
        %I = call double @sin( double %H )              ; <double> [#uses=1]
        ret double %I
}


; COS
declare float @cosf(float)

declare double @cos(double)

define double @test_cos(float %F) {
        %G = call float @cosf( float %F )               ; <float> [#uses=1]
        %H = fpext float %G to double           ; <double> [#uses=1]
        %I = call double @cos( double %H )              ; <double> [#uses=1]
        ret double %I
}

