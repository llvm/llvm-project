; RUN: not llvm-as < %s 2>&1 | grep "value doesn't match function result type 'i32'"

; Verify the operand type of the ret instructions in a function match the
; delcared return type of the function they live in.
;

define i32 @testfunc() {
	ret i32* null
}
