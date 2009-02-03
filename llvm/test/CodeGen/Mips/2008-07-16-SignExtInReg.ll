; RUN: llvm-as < %s | llc -march=mips -f -o %t
; RUN: grep seh %t | count 1
; RUN: grep seb %t | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define i8 @A(i8 %e.0, i8 signext %sum) signext nounwind {
entry:
	add i8 %sum, %e.0		; <i8>:0 [#uses=1]
	ret i8 %0
}

define i16 @B(i16 %e.0, i16 signext %sum) signext nounwind {
entry:
	add i16 %sum, %e.0		; <i16>:0 [#uses=1]
	ret i16 %0
}

