; Make sure this testcase codegens to the fabs instruction, not a call to fabsf
; RUN: llvm-as < %s | llc -march=x86 -mattr=-sse2,-sse3,-sse | grep fabs\$ | \
; RUN:   count 2
; RUN: llvm-as < %s | \
; RUN:   llc -march=x86 -mattr=-sse,-sse2,-sse3 -enable-unsafe-fp-math | \
; RUN:   grep fabs\$ | count 3

declare float @fabsf(float)

declare x86_fp80 @fabsl(x86_fp80)

define float @test1(float %X) {
        %Y = call float @fabsf(float %X)
        ret float %Y
}

define double @test2(double %X) {
        %Y = fcmp oge double %X, -0.0
        %Z = sub double -0.0, %X
        %Q = select i1 %Y, double %X, double %Z
        ret double %Q
}

define x86_fp80 @test3(x86_fp80 %X) {
        %Y = call x86_fp80 @fabsl(x86_fp80 %X)
        ret x86_fp80 %Y
}


