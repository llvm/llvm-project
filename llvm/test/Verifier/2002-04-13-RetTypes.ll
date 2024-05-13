; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Verify the operand type of the ret instructions in a function match the
; declared return type of the function they live in.
; CHECK: value doesn't match function result type 'i32'
;

define i32 @testfunc() {
	ret ptr null
}
