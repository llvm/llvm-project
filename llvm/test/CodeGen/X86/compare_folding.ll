; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | \
; RUN:   grep movsd | count 1
; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | \
; RUN:   grep ucomisd
declare i1 @llvm.isunordered.f64(double, double)

define i1 @test1(double %X, double %Y) {
        %COM = fcmp uno double %X, %Y           ; <i1> [#uses=1]
        ret i1 %COM
}

