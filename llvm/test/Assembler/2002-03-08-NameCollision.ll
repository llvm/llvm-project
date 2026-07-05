; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

; Method arguments were being checked for collisions at the global scope before
; the method object was created by the parser.  Because of this, false
; collisions could occur that would cause the following error message to be
; produced:
;
;    Redefinition of value named 'X' in the 'int *' type plane!
;
; Fixed by delaying binding of variable names until _after_ the method symtab is
; created.
;
@X = global i32 4		; <ptr> [#uses=0]

declare i32 @xxx(ptr)
