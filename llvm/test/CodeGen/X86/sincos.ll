; Make sure this testcase codegens to the sin and cos instructions, not calls
; RUN: llvm-as < %s | \
; RUN:   llc -march=x86 -mattr=-sse,-sse2,-sse3 -enable-unsafe-fp-math  | \
; RUN:   grep sin\$ | count 3
; RUN: llvm-as < %s | \
; RUN:   llc -march=x86 -mattr=-sse,-sse2,-sse3 -enable-unsafe-fp-math  | \
; RUN:   grep cos\$ | count 3

declare float  @sinf(float)

declare double @sin(double)

declare x86_fp80 @sinl(x86_fp80)

define float @test1(float %X) {
        %Y = call float @sinf(float %X)
        ret float %Y
}

define double @test2(double %X) {
        %Y = call double @sin(double %X)
        ret double %Y
}

define x86_fp80 @test3(x86_fp80 %X) {
        %Y = call x86_fp80 @sinl(x86_fp80 %X)
        ret x86_fp80 %Y
}

declare float @cosf(float)

declare double @cos(double)

declare x86_fp80 @cosl(x86_fp80)

define float @test4(float %X) {
        %Y = call float @cosf(float %X)
        ret float %Y
}

define double @test5(double %X) {
        %Y = call double @cos(double %X)
        ret double %Y
}

define x86_fp80 @test6(x86_fp80 %X) {
        %Y = call x86_fp80 @cosl(x86_fp80 %X)
        ret x86_fp80 %Y
}

