; RUN: llvm-as < %s | llc -march=mips -f -o %t
; RUN: grep abs.s  %t | count 1
; RUN: grep neg.s %t | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define float @A(float %i, float %j) nounwind  {
entry:
	tail call float @copysignf( float %i, float %j ) nounwind readnone 		; <float>:0 [#uses=1]
	ret float %0
}

declare float @copysignf(float, float) nounwind readnone 
