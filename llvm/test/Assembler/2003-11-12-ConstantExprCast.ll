; RUN: llvm-as < %s | llvm-dis | not grep " bitcast ("

@.Base64_1 = external constant [4 x i8]         ; <[4 x i8]*> [#uses=1]

define i8 @test(i8 %Y) {
        %X = bitcast i8 %Y to i8                ; <i8> [#uses=1]
        %tmp.13 = add i8 %X, sub (i8 0, i8 ptrtoint ([4 x i8]* @.Base64_1 to i8))     ; <i8> [#uses=1]
        ret i8 %tmp.13
}

