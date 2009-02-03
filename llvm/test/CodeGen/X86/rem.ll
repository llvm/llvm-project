; RUN: llvm-as < %s | llc -march=x86 | not grep div

define i32 @test1(i32 %X) {
        %tmp1 = srem i32 %X, 255                ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @test2(i32 %X) {
        %tmp1 = srem i32 %X, 256                ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @test3(i32 %X) {
        %tmp1 = urem i32 %X, 255                ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @test4(i32 %X) {
        %tmp1 = urem i32 %X, 256                ; <i32> [#uses=1]
        ret i32 %tmp1
}

