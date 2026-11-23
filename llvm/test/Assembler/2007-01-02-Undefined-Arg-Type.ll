; The assembler should catch an undefined argument type .
; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK: invalid type for function argument

; %typedef.bc_struct = type opaque


define i1 @someFunc(ptr %tmp.71.reload, %typedef.bc_struct %n1) {
	ret i1 true
}
