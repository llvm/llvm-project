; RUN: llc < %s -march=ppc32 -no-integrated-as -verify-machineinstrs | FileCheck %s

; CHECK-NOT: mr
define i32 @test(i32 %Y, i32 %X) {
entry:
        %tmp = tail call i32 asm "foo $0", "=r"( )              ; <i32> [#uses=1]
        ret i32 %tmp
}

define i32 @test2(i32 %Y, i32 %X) {
entry:
        %tmp1 = tail call i32 asm "foo $0, $1", "=r,r"( i32 %X )                ; <i32> [#uses=1]
        ret i32 %tmp1
}

; CHECK: test3
define i32 @test3(i32 %Y, i32 %X) {
entry:
        %tmp1 = tail call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "foo $0, $1", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"( i32 %X, i32 %Y, i32 %X, i32 %Y, i32 %X, i32 %Y, i32 %X, i32 %Y, i32 %X, i32 %Y, i32 %X, i32 %Y, i32 %X, i32 %Y, i32 %X, i32 %Y, i32 %X, i32 %Y, i32 %X, i32 %Y )                ; <i32> [#uses=1]
       ret i32 1
}
