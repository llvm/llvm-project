; RUN: llvm-as < %s | llc -march=mips -f -o %t
; RUN: grep mtc1 %t | count 1
; RUN: grep mfc1 %t | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define float @A(i32 %u) nounwind  {
entry:
	bitcast i32 %u to float
	ret float %0
}

define i32 @B(float %u) nounwind  {
entry:
	bitcast float %u to i32
	ret i32 %0
}
