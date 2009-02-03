; RUN: not llvm-as < %s |& grep {return type does not match operand type}

; Verify the the operand type of the ret instructions in a function match the
; delcared return type of the function they live in.
;

define i32 @testfunc()
begin
	ret i32* null
end
