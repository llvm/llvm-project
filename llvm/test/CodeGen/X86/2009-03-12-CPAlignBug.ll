; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=+sse2 | not grep ".space"
; rdar://6668548

declare double @llvm.sqrt.f64(double) nounwind readonly

declare double @fabs(double)

declare double @llvm.pow.f64(double, double) nounwind readonly

define void @SolveCubic_bb1(ptr %solutions, ptr %x, x86_fp80 %.reload, x86_fp80 %.reload5, x86_fp80 %.reload6, double %.reload8) nounwind {
newFuncRoot:
	br label %bb1

bb1.ret.exitStub:		; preds = %bb1
	ret void

bb1:		; preds = %newFuncRoot
	store i32 1, ptr %solutions, align 4
	%0 = tail call double @llvm.sqrt.f64(double %.reload8)		; <double> [#uses=1]
	%1 = fptrunc x86_fp80 %.reload6 to double		; <double> [#uses=1]
	%2 = tail call double @fabs(double %1) nounwind readnone		; <double> [#uses=1]
	%3 = fadd double %0, %2		; <double> [#uses=1]
	%4 = tail call double @llvm.pow.f64(double %3, double 0x3FD5555555555555)		; <double> [#uses=1]
	%5 = fpext double %4 to x86_fp80		; <x86_fp80> [#uses=2]
	%6 = fdiv x86_fp80 %.reload5, %5		; <x86_fp80> [#uses=1]
	%7 = fadd x86_fp80 %5, %6		; <x86_fp80> [#uses=1]
	%8 = fptrunc x86_fp80 %7 to double		; <double> [#uses=1]
	%9 = fcmp olt x86_fp80 %.reload6, 0xK00000000000000000000		; <i1> [#uses=1]
	%iftmp.6.0 = select i1 %9, double 1.000000e+00, double -1.000000e+00		; <double> [#uses=1]
	%10 = fmul double %8, %iftmp.6.0		; <double> [#uses=1]
	%11 = fpext double %10 to x86_fp80		; <x86_fp80> [#uses=1]
	%12 = fdiv x86_fp80 %.reload, 0xKC000C000000000000000		; <x86_fp80> [#uses=1]
	%13 = fadd x86_fp80 %11, %12		; <x86_fp80> [#uses=1]
	%14 = fptrunc x86_fp80 %13 to double		; <double> [#uses=1]
	store double %14, ptr %x, align 1
	br label %bb1.ret.exitStub
}
