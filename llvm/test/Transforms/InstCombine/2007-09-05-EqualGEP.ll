; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i1 true}
; PR1637

define i1 @f(i8* %arr) {
        %X = getelementptr i8* %arr, i32 1
        %Y = getelementptr i8* %arr, i32 1
        %test = icmp uge i8* %X, %Y
        ret i1 %test
}

