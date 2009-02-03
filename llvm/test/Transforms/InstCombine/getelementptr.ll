; The %A getelementptr instruction should be eliminated here

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep -v %B | not grep getelementptr
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep foo1
; END.

@Global = constant [10 x i8] c"helloworld"              ; <[10 x i8]*> [#uses=1]

; Test noop elimination
define i32* @foo1(i32* %I) {
        %A = getelementptr i32* %I, i64 0               ; <i32*> [#uses=1]
        ret i32* %A
}

; Test noop elimination
define i32* @foo2(i32* %I) {
        %A = getelementptr i32* %I              ; <i32*> [#uses=1]
        ret i32* %A
}

; Test that two array indexing geps fold
define i32* @foo3(i32* %I) {
        %A = getelementptr i32* %I, i64 17              ; <i32*> [#uses=1]
        %B = getelementptr i32* %A, i64 4               ; <i32*> [#uses=1]
        ret i32* %B
}

; Test that two getelementptr insts fold
define i32* @foo4({ i32 }* %I) {
        %A = getelementptr { i32 }* %I, i64 1           ; <{ i32 }*> [#uses=1]
        %B = getelementptr { i32 }* %A, i64 0, i32 0            ; <i32*> [#uses=1]
        ret i32* %B
}

define void @foo5(i8 %B) {
        ; This should be turned into a constexpr instead of being an instruction
        %A = getelementptr [10 x i8]* @Global, i64 0, i64 4             ; <i8*> [#uses=1]
        store i8 %B, i8* %A
        ret void
}

define i32* @foo6() {
        %M = malloc [4 x i32]           ; <[4 x i32]*> [#uses=1]
        %A = getelementptr [4 x i32]* %M, i64 0, i64 0          ; <i32*> [#uses=1]
        %B = getelementptr i32* %A, i64 2               ; <i32*> [#uses=1]
        ret i32* %B
}

define i32* @foo7(i32* %I, i64 %C, i64 %D) {
        %A = getelementptr i32* %I, i64 %C              ; <i32*> [#uses=1]
        %B = getelementptr i32* %A, i64 %D              ; <i32*> [#uses=1]
        ret i32* %B
}

define i8* @foo8([10 x i32]* %X) {
        ;; Fold into the cast.
        %A = getelementptr [10 x i32]* %X, i64 0, i64 0         ; <i32*> [#uses=1]
        %B = bitcast i32* %A to i8*             ; <i8*> [#uses=1]
        ret i8* %B
}

define i32 @test9() {
        %A = getelementptr { i32, double }* null, i32 0, i32 1          ; <double*> [#uses=1]
        %B = ptrtoint double* %A to i32         ; <i32> [#uses=1]
        ret i32 %B
}

define i1 @test10({ i32, i32 }* %x, { i32, i32 }* %y) {
        %tmp.1 = getelementptr { i32, i32 }* %x, i32 0, i32 1           ; <i32*> [#uses=1]
        %tmp.3 = getelementptr { i32, i32 }* %y, i32 0, i32 1           ; <i32*> [#uses=1]
        ;; seteq x, y
        %tmp.4 = icmp eq i32* %tmp.1, %tmp.3            ; <i1> [#uses=1]
        ret i1 %tmp.4
}

define i1 @test11({ i32, i32 }* %X) {
        %P = getelementptr { i32, i32 }* %X, i32 0, i32 0               ; <i32*> [#uses=1]
        %Q = icmp eq i32* %P, null              ; <i1> [#uses=1]
        ret i1 %Q
}
