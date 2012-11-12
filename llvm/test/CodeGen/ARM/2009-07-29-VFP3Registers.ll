; RUN: llc < %s -mtriple=armv7-apple-darwin10 -mattr=+vfp3

@a = external global double		; <double*> [#uses=1]

declare double @llvm.exp.f64(double) nounwind readonly

define void @findratio(double* nocapture %res1, double* nocapture %res2) nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	br i1 undef, label %bb28, label %bb

bb28:		; preds = %bb
	%0 = load double* @a, align 4		; <double> [#uses=2]
	%1 = fadd double %0, undef		; <double> [#uses=2]
	br i1 undef, label %bb59, label %bb60

bb59:		; preds = %bb28
	%2 = fsub double -0.000000e+00, undef		; <double> [#uses=2]
	br label %bb61

bb60:		; preds = %bb28
	%3 = tail call double @llvm.exp.f64(double undef) nounwind		; <double> [#uses=1]
	%4 = fsub double -0.000000e+00, %3		; <double> [#uses=2]
	%5 = fsub double -0.000000e+00, undef		; <double> [#uses=1]
	%6 = fsub double -0.000000e+00, undef		; <double> [#uses=1]
	br label %bb61

bb61:		; preds = %bb60, %bb59
	%.pn201 = phi double [ undef, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn111 = phi double [ undef, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn452 = phi double [ undef, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn85 = phi double [ undef, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn238 = phi double [ 0.000000e+00, %bb59 ], [ 0.000000e+00, %bb60 ]		; <double> [#uses=1]
	%.pn39 = phi double [ undef, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn230 = phi double [ undef, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn228 = phi double [ 0.000000e+00, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn224 = phi double [ undef, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn222 = phi double [ 0.000000e+00, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn218 = phi double [ %2, %bb59 ], [ %4, %bb60 ]		; <double> [#uses=1]
	%.pn214 = phi double [ 0.000000e+00, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn212 = phi double [ %2, %bb59 ], [ %4, %bb60 ]		; <double> [#uses=1]
	%.pn213 = phi double [ undef, %bb59 ], [ undef, %bb60 ]		; <double> [#uses=1]
	%.pn210 = phi double [ undef, %bb59 ], [ %5, %bb60 ]		; <double> [#uses=1]
	%.pn202 = phi double [ undef, %bb59 ], [ %6, %bb60 ]		; <double> [#uses=0]
	%.pn390 = fdiv double %.pn452, undef		; <double> [#uses=0]
	%.pn145 = fdiv double %.pn238, %1		; <double> [#uses=0]
	%.pn138 = fdiv double %.pn230, undef		; <double> [#uses=1]
	%.pn139 = fdiv double %.pn228, undef		; <double> [#uses=1]
	%.pn134 = fdiv double %.pn224, %0		; <double> [#uses=1]
	%.pn135 = fdiv double %.pn222, %1		; <double> [#uses=1]
	%.pn133 = fdiv double %.pn218, undef		; <double> [#uses=0]
	%.pn128 = fdiv double %.pn214, undef		; <double> [#uses=1]
	%.pn129 = fdiv double %.pn212, %.pn213		; <double> [#uses=1]
	%.pn126 = fdiv double %.pn210, undef		; <double> [#uses=0]
	%.pn54.in = fmul double undef, %.pn201		; <double> [#uses=1]
	%.pn42.in = fmul double undef, undef		; <double> [#uses=1]
	%.pn76 = fsub double %.pn138, %.pn139		; <double> [#uses=1]
	%.pn74 = fsub double %.pn134, %.pn135		; <double> [#uses=1]
	%.pn70 = fsub double %.pn128, %.pn129		; <double> [#uses=1]
	%.pn54 = fdiv double %.pn54.in, 6.000000e+00		; <double> [#uses=1]
	%.pn64 = fmul double undef, 0x3FE5555555555555		; <double> [#uses=1]
	%.pn65 = fmul double undef, undef		; <double> [#uses=1]
	%.pn50 = fmul double undef, %.pn111		; <double> [#uses=0]
	%.pn42 = fdiv double %.pn42.in, 6.000000e+00		; <double> [#uses=1]
	%.pn40 = fmul double undef, %.pn85		; <double> [#uses=0]
	%.pn56 = fadd double %.pn76, undef		; <double> [#uses=1]
	%.pn57 = fmul double %.pn74, undef		; <double> [#uses=1]
	%.pn36 = fadd double undef, undef		; <double> [#uses=1]
	%.pn37 = fmul double %.pn70, undef		; <double> [#uses=1]
	%.pn33 = fmul double undef, 0x3FC5555555555555		; <double> [#uses=1]
	%.pn29 = fsub double %.pn64, %.pn65		; <double> [#uses=1]
	%.pn21 = fadd double undef, undef		; <double> [#uses=1]
	%.pn27 = fmul double undef, 0x3FC5555555555555		; <double> [#uses=1]
	%.pn11 = fadd double %.pn56, %.pn57		; <double> [#uses=1]
	%.pn32 = fmul double %.pn54, undef		; <double> [#uses=1]
	%.pn26 = fmul double %.pn42, undef		; <double> [#uses=1]
	%.pn15 = fmul double 0.000000e+00, %.pn39		; <double> [#uses=1]
	%.pn7 = fadd double %.pn36, %.pn37		; <double> [#uses=1]
	%.pn30 = fsub double %.pn32, %.pn33		; <double> [#uses=1]
	%.pn28 = fadd double %.pn30, 0.000000e+00		; <double> [#uses=1]
	%.pn24 = fsub double %.pn28, %.pn29		; <double> [#uses=1]
	%.pn22 = fsub double %.pn26, %.pn27		; <double> [#uses=1]
	%.pn20 = fadd double %.pn24, undef		; <double> [#uses=1]
	%.pn18 = fadd double %.pn22, 0.000000e+00		; <double> [#uses=1]
	%.pn16 = fsub double %.pn20, %.pn21		; <double> [#uses=1]
	%.pn14 = fsub double %.pn18, undef		; <double> [#uses=1]
	%.pn12 = fadd double %.pn16, undef		; <double> [#uses=1]
	%.pn10 = fadd double %.pn14, %.pn15		; <double> [#uses=1]
	%.pn8 = fsub double %.pn12, undef		; <double> [#uses=1]
	%.pn6 = fsub double %.pn10, %.pn11		; <double> [#uses=1]
	%.pn4 = fadd double %.pn8, undef		; <double> [#uses=1]
	%.pn2 = fadd double %.pn6, %.pn7		; <double> [#uses=1]
	%N1.0 = fsub double %.pn4, undef		; <double> [#uses=1]
	%D1.0 = fsub double %.pn2, undef		; <double> [#uses=2]
	br i1 undef, label %bb62, label %bb64

bb62:		; preds = %bb61
	%7 = fadd double %D1.0, undef		; <double> [#uses=1]
	br label %bb64

bb64:		; preds = %bb62, %bb61
	%.pn = phi double [ undef, %bb62 ], [ %N1.0, %bb61 ]		; <double> [#uses=1]
	%.pn1 = phi double [ %7, %bb62 ], [ %D1.0, %bb61 ]		; <double> [#uses=1]
	%x.1 = fdiv double %.pn, %.pn1		; <double> [#uses=0]
	ret void
}
