; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i32 0}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep malloc
; PR1313

define i32 @test1(i32 %argc, i8* %argv, i8* %envp) {
        %tmp15.i.i.i23 = malloc [2564 x i32]            ; <[2564 x i32]*> [#uses=1]
        %c = icmp eq [2564 x i32]* %tmp15.i.i.i23, null              ; <i1>:0 [#uses=1]
        %retval = zext i1 %c to i32             ; <i32> [#uses=1]
        ret i32 %retval
}

define i32 @test2(i32 %argc, i8* %argv, i8* %envp) {
        %tmp15.i.i.i23 = malloc [2564 x i32]            ; <[2564 x i32]*> [#uses=1]
        %X = bitcast [2564 x i32]* %tmp15.i.i.i23 to i32*
        %c = icmp ne i32* %X, null
        %retval = zext i1 %c to i32             ; <i32> [#uses=1]
        ret i32 %retval
}

