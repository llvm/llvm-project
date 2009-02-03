; RUN: llvm-as < %s | opt -dse | llvm-dis | not grep DEAD

declare void @ext()

define i32* @caller() {
        %P = malloc i32         ; <i32*> [#uses=4]
        %DEAD = load i32* %P            ; <i32> [#uses=1]
        %DEAD2 = add i32 %DEAD, 1               ; <i32> [#uses=1]
        store i32 %DEAD2, i32* %P
        call void @ext( )
        store i32 0, i32* %P
        ret i32* %P
}

