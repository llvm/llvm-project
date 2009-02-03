; RUN: llvm-as < %s | opt -anders-aa
; PR3262

@.str15 = external global [3 x i8]              ; <[3 x i8]*> [#uses=1]

declare i8* @strtok(...)
declare i8* @memmove(...)

define void @test1(i8* %want1) nounwind {
entry:
        %0 = call i8* (...)* @strtok(i32 0, i8* getelementptr ([3 x i8]* @.str15, i32 0, i32 0)) nounwind               ; <i8*> [#uses=0]
        unreachable
}

define void @test2() nounwind {
entry:
        %0 = call i8* (...)* @memmove()
        unreachable
}
