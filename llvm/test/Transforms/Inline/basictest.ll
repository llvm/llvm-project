; RUN: llvm-as < %s | opt -inline -disable-output -print-function 2> /dev/null

define i32 @func(i32 %i) {
        ret i32 %i
}

define i32 @main(i32 %argc) {
        %X = call i32 @func( i32 7 )            ; <i32> [#uses=1]
        %Y = add i32 %X, %argc          ; <i32> [#uses=1]
        ret i32 %Y
}

