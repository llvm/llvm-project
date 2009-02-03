; RUN: llvm-as < %s | llc -march=mips | grep __adddf3

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define double @dofloat(double %a, double %b) nounwind {
entry:
	add double %a, %b		; <double>:0 [#uses=1]
	ret double %0
}
