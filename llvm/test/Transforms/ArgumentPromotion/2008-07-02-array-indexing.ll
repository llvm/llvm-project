; RUN: llvm-as < %s | opt -argpromotion | llvm-dis > %t
; RUN: cat %t | grep {define.*@callee(.*i32\\*}
; PR2498

; This test tries to convince argpromotion about promoting the load from %A + 2,
; because there is a load of %A in the entry block
define internal i32 @callee(i1 %C, i32* %A) {
entry:
        ; Unconditonally load the element at %A
        %A.0 = load i32* %A
        br i1 %C, label %T, label %F
T:
        ret i32 %A.0
F:
        ; Load the element at offset two from %A. This should not be promoted!
        %A.2 = getelementptr i32* %A, i32 2
        %R = load i32* %A.2
        ret i32 %R
}

define i32 @foo() {
        %X = call i32 @callee(i1 false, i32* null)             ; <i32> [#uses=1]
        ret i32 %X
}

