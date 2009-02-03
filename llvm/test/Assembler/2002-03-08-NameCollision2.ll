; RUN: llvm-as %s -o /dev/null -f

; Another name collision problem.  Here the problem was that if a forward
; declaration for a method was found, that this would cause spurious conflicts
; to be detected between locals and globals.
;
@Var = external global i32		; <i32*> [#uses=0]

define void @foo() {
	%Var = alloca i32		; <i32*> [#uses=0]
	ret void
}
