; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -pre-alloc-split

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	br label %bb14.i

bb14.i:		; preds = %bb14.i, %entry
	%i8.0.reg2mem.0.i = phi i32 [ 0, %entry ], [ %0, %bb14.i ]		; <i32> [#uses=1]
	%0 = add i32 %i8.0.reg2mem.0.i, 1		; <i32> [#uses=2]
	%1 = add double 0.000000e+00, 0.000000e+00		; <double> [#uses=1]
	%2 = add double 0.000000e+00, 0.000000e+00		; <double> [#uses=1]
	%3 = add double 0.000000e+00, 0.000000e+00		; <double> [#uses=1]
	%exitcond75.i = icmp eq i32 %0, 32		; <i1> [#uses=1]
	br i1 %exitcond75.i, label %bb24.i, label %bb14.i

bb24.i:		; preds = %bb14.i
	%4 = fdiv double 0.000000e+00, 0.000000e+00		; <double> [#uses=1]
	%5 = fdiv double %1, 0.000000e+00		; <double> [#uses=1]
	%6 = fdiv double %2, 0.000000e+00		; <double> [#uses=1]
	%7 = fdiv double %3, 0.000000e+00		; <double> [#uses=1]
	br label %bb31.i

bb31.i:		; preds = %bb31.i, %bb24.i
	%tmp.0.reg2mem.0.i = phi i32 [ 0, %bb24.i ], [ %indvar.next64.i, %bb31.i ]		; <i32> [#uses=1]
	%indvar.next64.i = add i32 %tmp.0.reg2mem.0.i, 1		; <i32> [#uses=2]
	%exitcond65.i = icmp eq i32 %indvar.next64.i, 64		; <i1> [#uses=1]
	br i1 %exitcond65.i, label %bb33.i, label %bb31.i

bb33.i:		; preds = %bb31.i
	br label %bb35.preheader.i

bb5.i.i:		; preds = %bb35.preheader.i
	%8 = call double @floor(double 0.000000e+00) nounwind readnone		; <double> [#uses=0]
	br label %bb7.i.i

bb7.i.i:		; preds = %bb35.preheader.i, %bb5.i.i
	br label %bb35.preheader.i

bb35.preheader.i:		; preds = %bb7.i.i, %bb33.i
	%9 = sub double 0.000000e+00, %4		; <double> [#uses=1]
	store double %9, double* null, align 8
	%10 = sub double 0.000000e+00, %5		; <double> [#uses=1]
	store double %10, double* null, align 8
	%11 = sub double 0.000000e+00, %6		; <double> [#uses=1]
	store double %11, double* null, align 8
	%12 = sub double 0.000000e+00, %7		; <double> [#uses=1]
	store double %12, double* null, align 8
	br i1 false, label %bb7.i.i, label %bb5.i.i
}

declare double @floor(double) nounwind readnone
