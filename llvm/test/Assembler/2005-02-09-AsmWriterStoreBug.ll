; RUN: llvm-as < %s | llvm-dis | llvm-as

; Ensure that the asm writer emits types before both operands of the 
; store, even though they can be the same.

%RecTy = type %RecTy*

define void @foo() {
        %A = malloc %RecTy              ; <%RecTy> [#uses=1]
        %B = malloc %RecTy              ; <%RecTy> [#uses=1]
        store %RecTy %B, %RecTy %A
        ret void
}

