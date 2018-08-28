; RUN: llc -verify-machineinstrs < %s
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-unknown-linux-gnu"

define double @SolveCubic(ppc_fp128 %X) {
entry:
	%Y = fptrunc ppc_fp128 %X to double
	ret double %Y
}
