; The assembler should catch an undefined argument type .
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: use of undefined type named 'typedef.bc_struct'

; %typedef.bc_struct = type opaque


define i1 @someFunc(ptr %tmp.71.reload, %typedef.bc_struct %n1) {
	ret i1 true
}
