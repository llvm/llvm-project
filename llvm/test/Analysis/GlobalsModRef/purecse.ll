; Test that pure functions are cse'd away
; RUN: opt < %s -globals-aa -gvn -instcombine -S | FileCheck %s

define i32 @pure(i32 %X) {
        %Y = add i32 %X, 1              ; <i32> [#uses=1]
        ret i32 %Y
}

define i32 @test1(i32 %X) {
; CHECK:      %A = call i32 @pure(i32 %X)
; CHECK-NEXT: ret i32 0
        %A = call i32 @pure( i32 %X )           ; <i32> [#uses=1]
        %B = call i32 @pure( i32 %X )           ; <i32> [#uses=1]
        %C = sub i32 %A, %B             ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test2(i32 %X, i32* %P) {
; CHECK:      %A = call i32 @pure(i32 %X)
; CHECK-NEXT: store i32 %X, i32* %P
; CHECK-NEXT: ret i32 0
        %A = call i32 @pure( i32 %X )           ; <i32> [#uses=1]
        store i32 %X, i32* %P ;; Does not invalidate 'pure' call.
        %B = call i32 @pure( i32 %X )           ; <i32> [#uses=1]
        %C = sub i32 %A, %B             ; <i32> [#uses=1]
        ret i32 %C
}
