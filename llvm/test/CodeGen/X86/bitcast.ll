; RUN: llvm-as < %s | llc -march=x86
; RUN: llvm-as < %s | llc -march=x86-64
; PR1033

define i64 @test1(double %t) {
        %u = bitcast double %t to i64           ; <i64> [#uses=1]
        ret i64 %u
}

define double @test2(i64 %t) {
        %u = bitcast i64 %t to double           ; <double> [#uses=1]
        ret double %u
}

define i32 @test3(float %t) {
        %u = bitcast float %t to i32            ; <i32> [#uses=1]
        ret i32 %u
}

define float @test4(i32 %t) {
        %u = bitcast i32 %t to float            ; <float> [#uses=1]
        ret float %u
}

