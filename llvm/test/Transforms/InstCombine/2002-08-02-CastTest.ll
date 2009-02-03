; This testcase is incorrectly getting completely eliminated.  There should be
; SOME instruction named %c here, even if it's a bitwise and.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep %c
;
define i64 @test3(i64 %A) {
        %c1 = trunc i64 %A to i8                ; <i8> [#uses=1]
        %c2 = zext i8 %c1 to i64                ; <i64> [#uses=1]
        ret i64 %c2
}

