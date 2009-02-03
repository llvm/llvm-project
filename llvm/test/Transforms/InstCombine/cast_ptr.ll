; Tests to make sure elimination of casts is working correctly
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | notcast

target datalayout = "p:32:32"

define i8* @test1(i8* %t) {
        %tmpc = ptrtoint i8* %t to i32          ; <i32> [#uses=1]
        %tmpa = add i32 %tmpc, 32               ; <i32> [#uses=1]
        %tv = inttoptr i32 %tmpa to i8*         ; <i8*> [#uses=1]
        ret i8* %tv
}

define i1 @test2(i8* %a, i8* %b) {
        %tmpa = ptrtoint i8* %a to i32          ; <i32> [#uses=1]
        %tmpb = ptrtoint i8* %b to i32          ; <i32> [#uses=1]
        %r = icmp eq i32 %tmpa, %tmpb           ; <i1> [#uses=1]
        ret i1 %r
}

