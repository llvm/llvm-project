; RUN: llc < %s -mtriple=i686--

	%struct.CompAtom = type <{ %struct.Position, float, i32 }>
	%struct.Lattice = type { %struct.Position, %struct.Position, %struct.Position, %struct.Position, %struct.Position, %struct.Position, %struct.Position, i32, i32, i32 }
	%struct.Position = type { double, double, double }

define fastcc ptr @_ZNK7Lattice6createEP8CompAtomii(ptr %this, ptr %d, i32 %n, i32 %i) {
entry:
	%tmp18 = tail call ptr @_Znam( i32 0 )		; <ptr> [#uses=1]
	%tmp3286 = icmp eq i32 %n, 0		; <i1> [#uses=1]
	br i1 %tmp3286, label %bb35, label %bb24

bb24:		; preds = %bb24, %entry
	%tmp9.0.reg2mem.0.rec = phi i32 [ %indvar.next, %bb24 ], [ 0, %entry ]		; <i32> [#uses=3]
	%tmp3.i.i = getelementptr %struct.CompAtom, ptr %tmp18, i32 %tmp9.0.reg2mem.0.rec, i32 0, i32 1		; <ptr> [#uses=0]
	%tmp5.i.i = getelementptr %struct.CompAtom, ptr %tmp18, i32 %tmp9.0.reg2mem.0.rec, i32 0, i32 2		; <ptr> [#uses=1]
	store double -9.999900e+04, ptr %tmp5.i.i, align 4
	%indvar.next = add i32 %tmp9.0.reg2mem.0.rec, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %n		; <i1> [#uses=1]
	br i1 %exitcond, label %bb35, label %bb24

bb35:		; preds = %bb24, %entry
	%tmp42 = sdiv i32 %i, 9		; <i32> [#uses=1]
	%tmp43 = add i32 %tmp42, -1		; <i32> [#uses=1]
	%tmp4344 = sitofp i32 %tmp43 to double		; <double> [#uses=1]
	%tmp17.i76 = fmul double %tmp4344, 0.000000e+00		; <double> [#uses=1]
	%tmp48 = sdiv i32 %i, 3		; <i32> [#uses=1]
	%tmp49 = srem i32 %tmp48, 3		; <i32> [#uses=1]
	%tmp50 = add i32 %tmp49, -1		; <i32> [#uses=1]
	%tmp5051 = sitofp i32 %tmp50 to double		; <double> [#uses=1]
	%tmp17.i63 = fmul double %tmp5051, 0.000000e+00		; <double> [#uses=1]
	%tmp55 = srem i32 %i, 3		; <i32> [#uses=1]
	%tmp56 = add i32 %tmp55, -1		; <i32> [#uses=1]
	%tmp5657 = sitofp i32 %tmp56 to double		; <double> [#uses=1]
	%tmp15.i49 = getelementptr %struct.Lattice, ptr %this, i32 0, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp16.i50 = load double, ptr %tmp15.i49, align 4		; <double> [#uses=1]
	%tmp17.i = fmul double %tmp5657, %tmp16.i50		; <double> [#uses=1]
	%tmp20.i39 = fadd double %tmp17.i, %tmp17.i63		; <double> [#uses=1]
	%tmp20.i23 = fadd double %tmp20.i39, %tmp17.i76		; <double> [#uses=1]
	br i1 false, label %bb58.preheader, label %bb81

bb58.preheader:		; preds = %bb35
	%smax = select i1 false, i32 1, i32 %n		; <i32> [#uses=1]
	br label %bb58

bb58:		; preds = %bb58, %bb58.preheader
	%tmp20.i7 = getelementptr %struct.CompAtom, ptr %d, i32 0, i32 2		; <ptr> [#uses=2]
	%tmp25.i = getelementptr %struct.CompAtom, ptr %tmp18, i32 0, i32 2		; <ptr> [#uses=2]
	%tmp74.i = load i32, ptr %tmp20.i7, align 1		; <i32> [#uses=1]
	%tmp82.i = and i32 %tmp74.i, 134217728		; <i32> [#uses=1]
	%tmp85.i = or i32 0, %tmp82.i		; <i32> [#uses=1]
	store i32 %tmp85.i, ptr %tmp25.i, align 1
	%tmp88.i = load i32, ptr %tmp20.i7, align 1		; <i32> [#uses=1]
	%tmp95.i = and i32 %tmp88.i, -268435456		; <i32> [#uses=1]
	%tmp97.i = or i32 0, %tmp95.i		; <i32> [#uses=1]
	store i32 %tmp97.i, ptr %tmp25.i, align 1
	%tmp6.i = fadd double 0.000000e+00, %tmp20.i23		; <double> [#uses=0]
	%exitcond96 = icmp eq i32 0, %smax		; <i1> [#uses=1]
	br i1 %exitcond96, label %bb81, label %bb58

bb81:		; preds = %bb58, %bb35
	ret ptr %tmp18
}

declare ptr @_Znam(i32)
