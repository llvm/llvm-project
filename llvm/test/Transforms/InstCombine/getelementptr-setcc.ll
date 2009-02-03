; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep getelementptr

define i1 @test1(i16* %P, i32 %I, i32 %J) {
        %X = getelementptr i16* %P, i32 %I              ; <i16*> [#uses=1]
        %Y = getelementptr i16* %P, i32 %J              ; <i16*> [#uses=1]
        %C = icmp ult i16* %X, %Y               ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test2(i16* %P, i32 %I) {
        %X = getelementptr i16* %P, i32 %I              ; <i16*> [#uses=1]
        %C = icmp ult i16* %X, %P               ; <i1> [#uses=1]
        ret i1 %C
}

define i32 @test3(i32* %P, i32 %A, i32 %B) {
        %tmp.4 = getelementptr i32* %P, i32 %A          ; <i32*> [#uses=1]
        %tmp.9 = getelementptr i32* %P, i32 %B          ; <i32*> [#uses=1]
        %tmp.10 = icmp eq i32* %tmp.4, %tmp.9           ; <i1> [#uses=1]
        %tmp.11 = zext i1 %tmp.10 to i32                ; <i32> [#uses=1]
        ret i32 %tmp.11
}

define i32 @test4(i32* %P, i32 %A, i32 %B) {
        %tmp.4 = getelementptr i32* %P, i32 %A          ; <i32*> [#uses=1]
        %tmp.6 = icmp eq i32* %tmp.4, %P                ; <i1> [#uses=1]
        %tmp.7 = zext i1 %tmp.6 to i32          ; <i32> [#uses=1]
        ret i32 %tmp.7
}

