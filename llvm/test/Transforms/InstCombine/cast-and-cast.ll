; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep bitcast

define i1 @test1(i32 %val) {
        %t1 = bitcast i32 %val to i32           ; <i32> [#uses=1]
        %t2 = and i32 %t1, 1            ; <i32> [#uses=1]
        %t3 = trunc i32 %t2 to i1               ; <i1> [#uses=1]
        ret i1 %t3
}

define i16 @test1.upgrd.1(i32 %val) {
        %t1 = bitcast i32 %val to i32           ; <i32> [#uses=1]
        %t2 = and i32 %t1, 1            ; <i32> [#uses=1]
        %t3 = trunc i32 %t2 to i16              ; <i16> [#uses=1]
        ret i16 %t3
}

