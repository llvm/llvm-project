; RUN: llc < %s -march=x86 | not grep call

declare i1 @llvm.isunordered.f64(double)

define i1 @test_isnan(double %X) {
        %R = fcmp uno double %X, %X             ; <i1> [#uses=1]
        ret i1 %R
}

