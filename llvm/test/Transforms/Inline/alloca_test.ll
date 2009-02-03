; This test ensures that alloca instructions in the entry block for an inlined
; function are moved to the top of the function they are inlined into.
;
; RUN: llvm-as < %s | opt -inline | llvm-dis | %prcontext alloca 1 | grep Entry:

define i32 @func(i32 %i) {
        %X = alloca i32         ; <i32*> [#uses=1]
        store i32 %i, i32* %X
        ret i32 %i
}

declare void @bar()

define i32 @main(i32 %argc) {
Entry:
        call void @bar( )
        %X = call i32 @func( i32 7 )            ; <i32> [#uses=1]
        %Y = add i32 %X, %argc          ; <i32> [#uses=1]
        ret i32 %Y
}

