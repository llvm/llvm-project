; This test makes sure that these instructions are properly eliminated.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep select


define i41 @test1(i1 %C) {
	%V = select i1 %C, i41 1, i41 0  ; V = C
	ret i41 %V
}

define i999 @test2(i1 %C) {
	%V = select i1 %C, i999 0, i999 1  ; V = C
	ret i999 %V
}

define i41 @test3(i41 %X) {
    ;; (x <s 0) ? -1 : 0 -> ashr x, 31
    %t = icmp slt i41 %X, 0
    %V = select i1 %t, i41 -1, i41 0
    ret i41 %V
}

define i1023 @test4(i1023 %X) {
    ;; (x <s 0) ? -1 : 0 -> ashr x, 31
    %t = icmp slt i1023 %X, 0
    %V = select i1 %t, i1023 -1, i1023 0
    ret i1023 %V
}

define i41 @test5(i41 %X) {
    ;; ((X & 27) ? 27 : 0)
    %Y = and i41 %X, 32
    %t = icmp ne i41 %Y, 0
    %V = select i1 %t, i41 32, i41 0
    ret i41 %V
}

define i1023 @test6(i1023 %X) {
    ;; ((X & 27) ? 27 : 0)
    %Y = and i1023 %X, 64 
    %t = icmp ne i1023 %Y, 0
    %V = select i1 %t, i1023 64, i1023 0
    ret i1023 %V
}
