; RUN: llvm-as < %s | opt -instcombine | llvm-dis | notcast
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {icmp s}
; PR1940

define i1 @test1(i8 %A, i8 %B) {
        %a = zext i8 %A to i32
        %b = zext i8 %B to i32
        %c = icmp sgt i32 %a, %b
        ret i1 %c
}

define i1 @test2(i8 %A, i8 %B) {
        %a = sext i8 %A to i32
        %b = sext i8 %B to i32
        %c = icmp ugt i32 %a, %b
        ret i1 %c
}
