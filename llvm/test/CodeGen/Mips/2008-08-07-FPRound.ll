; RUN: llvm-as < %s | llc -march=mips | grep __truncdfsf2 | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define float @round2float(double %a) nounwind {
entry:
	fptrunc double %a to float		; <float>:0 [#uses=1]
	ret float %0
}
