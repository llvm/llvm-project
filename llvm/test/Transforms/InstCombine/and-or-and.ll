; If we have an 'and' of the result of an 'or', and one of the 'or' operands
; cannot have contributed any of the resultant bits, delete the or.  This
; occurs for very common C/C++ code like this:
;
; struct foo { int A : 16; int B : 16; };
; void test(struct foo *F, int X, int Y) {
;        F->A = X; F->B = Y;
; }
;
; Which corresponds to test1.

; RUN: opt < %s -instcombine -S | \
; RUN:   not grep "or "

define i32 @test1(i32 %X, i32 %Y) {
        %A = and i32 %X, 7              ; <i32> [#uses=1]
        %B = and i32 %Y, 8              ; <i32> [#uses=1]
        %C = or i32 %A, %B              ; <i32> [#uses=1]
        ;; This cannot include any bits from %Y!
        %D = and i32 %C, 7              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test2(i32 %X, i8 %Y) {
        %B = zext i8 %Y to i32          ; <i32> [#uses=1]
        %C = or i32 %X, %B              ; <i32> [#uses=1]
        ;; This cannot include any bits from %Y!
        %D = and i32 %C, 65536          ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test3(i32 %X, i32 %Y) {
        %B = shl i32 %Y, 1              ; <i32> [#uses=1]
        %C = or i32 %X, %B              ; <i32> [#uses=1]
        ;; This cannot include any bits from %Y!
        %D = and i32 %C, 1              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test4(i32 %X, i32 %Y) {
        %B = lshr i32 %Y, 31            ; <i32> [#uses=1]
        %C = or i32 %X, %B              ; <i32> [#uses=1]
        ;; This cannot include any bits from %Y!
        %D = and i32 %C, 2              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @or_test1(i32 %X, i32 %Y) {
        %A = and i32 %X, 1              ; <i32> [#uses=1]
        ;; This cannot include any bits from X!
        %B = or i32 %A, 1               ; <i32> [#uses=1]
        ret i32 %B
}

define i8 @or_test2(i8 %X, i8 %Y) {
        %A = shl i8 %X, 7               ; <i8> [#uses=1]
        ;; This cannot include any bits from X!
        %B = or i8 %A, -128             ; <i8> [#uses=1]
        ret i8 %B
}

