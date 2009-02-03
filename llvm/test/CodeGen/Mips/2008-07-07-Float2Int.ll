; RUN: llvm-as < %s | llc -march=mips | grep trunc.w.s | count 3

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define i32 @fptoint(float %a) nounwind {
entry:
	fptosi float %a to i32		; <i32>:0 [#uses=1]
	ret i32 %0
}

define i32 @fptouint(float %a) nounwind {
entry:
	fptoui float %a to i32		; <i32>:0 [#uses=1]
	ret i32 %0
}
