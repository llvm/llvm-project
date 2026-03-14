; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll



declare void @bar()

define i9 @foo() {  ;; Calling this function has undefined behavior
	unreachable
}

define double @xyz() {
	call void @bar()
	unreachable          ;; Bar must not return.
}
