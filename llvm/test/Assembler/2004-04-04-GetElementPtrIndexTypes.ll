; RUN: llvm-as %s -o /dev/null -f

define i32* @t1({ float, i32 }* %X) {
        %W = getelementptr { float, i32 }* %X, i32 20, i32 1            ; <i32*> [#uses=0]
        %X.upgrd.1 = getelementptr { float, i32 }* %X, i64 20, i32 1            ; <i32*> [#uses=0]
        %Y = getelementptr { float, i32 }* %X, i64 20, i32 1            ; <i32*> [#uses=1]
        %Z = getelementptr { float, i32 }* %X, i64 20, i32 1            ; <i32*> [#uses=0]
        ret i32* %Y
}

