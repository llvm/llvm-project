; RUN: llc < %s
; RUN: llc -O0 < %s

declare float @llvm.sqrt.f32(float)

declare double @llvm.sqrt.f64(double)

define double @test_sqrt(float %F) {
        %G = call float @llvm.sqrt.f32( float %F )              ; <float> [#uses=1]
        %H = fpext float %G to double           ; <double> [#uses=1]
        %I = call double @llvm.sqrt.f64( double %H )            ; <double> [#uses=1]
        ret double %I
}

declare ptr @llvm.launder.invariant.group(ptr)

define ptr @launder(ptr %p) {
        %q = call ptr @llvm.launder.invariant.group(ptr %p)
        ret ptr %q
}

declare ptr @llvm.strip.invariant.group(ptr)

define ptr @strip(ptr %p) {
        %q = call ptr @llvm.strip.invariant.group(ptr %p)
        ret ptr %q
}

declare void @llvm.sideeffect()

define void @test_sideeffect() {
    call void @llvm.sideeffect()
    ret void
}
