; RUN: echo {@X = linkonce global i32 5 \
; RUN:   define linkonce i32 @foo() \{ ret i32 7 \} } | llvm-as > %t.1.bc
; RUN: llvm-as %s -o %t.2.bc -f
; RUN: llvm-link %t.1.bc  %t.2.bc
@X = external global i32 

declare i32 @foo() 

define void @bar() {
	load i32* @X
	call i32 @foo()
	ret void
}

