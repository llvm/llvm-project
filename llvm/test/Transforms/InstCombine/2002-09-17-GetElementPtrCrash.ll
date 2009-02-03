; RUN: llvm-as < %s | opt -instcombine

        %bob = type { i32 }

define i32 @alias() {
        %pbob1 = alloca %bob            ; <%bob*> [#uses=1]
        %pbob2 = getelementptr %bob* %pbob1             ; <%bob*> [#uses=1]
        %pbobel = getelementptr %bob* %pbob2, i64 0, i32 0              ; <i32*> [#uses=1]
        %rval = load i32* %pbobel               ; <i32> [#uses=1]
        ret i32 %rval
}

