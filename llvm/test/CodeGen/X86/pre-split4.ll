; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -pre-alloc-split -stats |& \
; RUN:   grep {pre-alloc-split} | grep {Number of intervals split} | grep 4

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%k.0.reg2mem.0 = phi double [ 1.000000e+00, %entry ], [ %6, %bb ]		; <double> [#uses=2]
	%Flint.0.reg2mem.0 = phi double [ 0.000000e+00, %entry ], [ %5, %bb ]		; <double> [#uses=1]
	%twoThrd.0.reg2mem.0 = phi double [ 0.000000e+00, %entry ], [ %1, %bb ]		; <double> [#uses=1]
	%0 = tail call double @llvm.pow.f64(double 0x3FE5555555555555, double 0.000000e+00)		; <double> [#uses=1]
	%1 = add double %0, %twoThrd.0.reg2mem.0		; <double> [#uses=1]
	%2 = tail call double @sin(double %k.0.reg2mem.0) nounwind readonly		; <double> [#uses=1]
	%3 = mul double 0.000000e+00, %2		; <double> [#uses=1]
	%4 = fdiv double 1.000000e+00, %3		; <double> [#uses=1]
	%5 = add double %4, %Flint.0.reg2mem.0		; <double> [#uses=1]
	%6 = add double %k.0.reg2mem.0, 1.000000e+00		; <double> [#uses=1]
	br label %bb
}

declare double @llvm.pow.f64(double, double) nounwind readonly

declare double @sin(double) nounwind readonly
