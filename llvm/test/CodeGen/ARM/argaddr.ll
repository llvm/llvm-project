; RUN: llc < %s -march=arm

define void @f(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
        %a_addr = alloca i32            ; <i32*> [#uses=2]
        %b_addr = alloca i32            ; <i32*> [#uses=2]
        %c_addr = alloca i32            ; <i32*> [#uses=2]
        %d_addr = alloca i32            ; <i32*> [#uses=2]
        %e_addr = alloca i32            ; <i32*> [#uses=2]
        store i32 %a, i32* %a_addr
        store i32 %b, i32* %b_addr
        store i32 %c, i32* %c_addr
        store i32 %d, i32* %d_addr
        store i32 %e, i32* %e_addr
        call void @g( i32* %a_addr, i32* %b_addr, i32* %c_addr, i32* %d_addr, i32* %e_addr )
        ret void
}

declare void @g(i32*, i32*, i32*, i32*, i32*)
