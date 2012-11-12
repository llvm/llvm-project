; RUN: llc < %s -mtriple=powerpc64-apple-darwin | grep extsw | count 2

@lens = external global i8*             ; <i8**> [#uses=1]
@vals = external global i32*            ; <i32**> [#uses=1]

define i32 @test(i32 %i) {
        %tmp = load i8** @lens          ; <i8*> [#uses=1]
        %tmp1 = getelementptr i8* %tmp, i32 %i          ; <i8*> [#uses=1]
        %tmp.upgrd.1 = load i8* %tmp1           ; <i8> [#uses=1]
        %tmp2 = zext i8 %tmp.upgrd.1 to i32             ; <i32> [#uses=1]
        %tmp3 = load i32** @vals                ; <i32*> [#uses=1]
        %tmp5 = sub i32 1, %tmp2                ; <i32> [#uses=1]
        %tmp6 = getelementptr i32* %tmp3, i32 %tmp5             ; <i32*> [#uses=1]
        %tmp7 = load i32* %tmp6         ; <i32> [#uses=1]
        ret i32 %tmp7
}

