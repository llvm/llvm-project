; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    notcast {} {getelementptr.*}

@G = external global [3 x i8]           ; <[3 x i8]*> [#uses=1]

define i8* @foo(i32 %Idx) {
        %gep.upgrd.1 = zext i32 %Idx to i64             ; <i64> [#uses=1]
        %tmp = getelementptr i8* getelementptr ([3 x i8]* @G, i32 0, i32 0), i64 %gep.upgrd.1              ; <i8*> [#uses=1]
        ret i8* %tmp
}

