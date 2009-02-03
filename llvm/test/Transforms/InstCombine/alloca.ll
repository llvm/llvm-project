; Zero byte allocas should be deleted.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep alloca
; END.

declare void @use(...)

define void @test() {
        %X = alloca [0 x i32]           ; <[0 x i32]*> [#uses=1]
        call void (...)* @use( [0 x i32]* %X )
        %Y = alloca i32, i32 0          ; <i32*> [#uses=1]
        call void (...)* @use( i32* %Y )
        %Z = alloca {  }                ; <{  }*> [#uses=1]
        call void (...)* @use( {  }* %Z )
        ret void
}

define void @test2() {
        %A = alloca i32         ; <i32*> [#uses=1]
        store i32 123, i32* %A
        ret void
}

define void @test3() {
        %A = alloca { i32 }             ; <{ i32 }*> [#uses=1]
        %B = getelementptr { i32 }* %A, i32 0, i32 0            ; <i32*> [#uses=1]
        store i32 123, i32* %B
        ret void
}

