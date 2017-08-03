;; X's live range extends beyond the shift, so the register allocator
;; cannot coalesce it with Y.  Because of this, a copy needs to be
;; emitted before the shift to save the register value before it is
;; clobbered.  However, this copy is not needed if the register
;; allocator turns the shift into an LEA.  This also occurs for ADD.

; Check that the shift gets turned into an LEA.

; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | \
; RUN:   not grep "mov E.X, E.X"

@G = external global i32                ; <i32*> [#uses=1]

define i32 @test1(i32 %X) {
        %Z = shl i32 %X, 2              ; <i32> [#uses=1]
        store volatile i32 %Z, i32* @G
        ret i32 %X
}

