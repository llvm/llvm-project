; Test that linking two files with the same definition causes an error and
; that error is printed out.
; RUN: llvm-as %s -o %t.one.bc -f
; RUN: llvm-as %s -o %t.two.bc -f
; RUN: not llvm-ld -disable-opt -link-as-library %t.one.bc %t.two.bc \
; RUN:   -o %t.bc 2>%t.err 
; RUN: grep "symbol multiply defined" %t.err

define i32 @bar() {
	ret i32 0
}
