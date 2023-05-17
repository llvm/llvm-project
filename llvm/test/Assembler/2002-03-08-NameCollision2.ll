; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

; Another name collision problem.  Here the problem was that if a forward
; declaration for a method was found, that this would cause spurious conflicts
; to be detected between locals and globals.
;
@Var = external global i32		; <ptr> [#uses=0]

define void @foo() {
	%Var = alloca i32		; <ptr> [#uses=0]
	ret void
}
