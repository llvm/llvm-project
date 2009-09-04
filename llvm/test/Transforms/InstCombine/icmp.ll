; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep icmp

define i32 @test1(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

define i32 @test2(i32 %X) {
entry:
        icmp ult i32 %X, -2147483648            ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

define i32 @test3(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        sext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

define i32 @test4(i32 %X) {
entry:
        icmp ult i32 %X, -2147483648            ; <i1>:0 [#uses=1]
        sext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

; PR4837
define <2 x i1> @test5(<2 x i64> %x) {
entry:
  %V = icmp eq <2 x i64> %x, undef
  ret <2 x i1> %V
}