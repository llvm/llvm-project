; A very rudimentary test on AliasAnalysis::getModRefInfo.
; RUN: llvm-as < %s | opt -print-all-alias-modref-info -aa-eval -disable-output |& \
; RUN: not grep NoModRef

define i32 @callee() {
        %X = alloca { i32, i32 }                ; <{ i32, i32 }*> [#uses=1]
        %Y = getelementptr { i32, i32 }* %X, i64 0, i32 0               ; <i32*> [#uses=1]
        %Z = load i32* %Y               ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @caller() {
        %X = call i32 @callee( )                ; <i32> [#uses=1]
        ret i32 %X
}
