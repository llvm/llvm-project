; RUN: llvm-as < %s | opt -inline -disable-output
	%struct.Demand = type { double, double }
	%struct.branch = type { %struct.Demand, double, double, double, double, %struct.branch*, [12 x %struct.leaf*] }
	%struct.leaf = type { %struct.Demand, double, double }
@P = external global double		; <double*> [#uses=1]

define %struct.leaf* @build_leaf() nounwind  {
entry:
	unreachable
}

define %struct.Demand @Compute_Branch2(%struct.branch* %br, double %theta_R, double %theta_I, double %pi_R, double %pi_I) nounwind  {
entry:
	%mrv = alloca %struct.Demand		; <%struct.Demand*> [#uses=4]
	%a2 = alloca %struct.Demand		; <%struct.Demand*> [#uses=0]
	br i1 false, label %bb46, label %bb
bb:		; preds = %entry
	%mrv.gep = getelementptr %struct.Demand* %mrv, i32 0, i32 0		; <double*> [#uses=1]
	%mrv.ld = load double* %mrv.gep		; <double> [#uses=1]
	%mrv.gep1 = getelementptr %struct.Demand* %mrv, i32 0, i32 1		; <double*> [#uses=1]
	%mrv.ld2 = load double* %mrv.gep1		; <double> [#uses=1]
	ret double %mrv.ld, double %mrv.ld2
bb46:		; preds = %entry
	br label %bb72
bb49:		; preds = %bb72
	call %struct.Demand @Compute_Leaf1( %struct.leaf* null, double 0.000000e+00, double 0.000000e+00 ) nounwind 		; <%struct.Demand>:0 [#uses=1]
	%gr = getresult %struct.Demand %0, 1		; <double> [#uses=0]
	br label %bb72
bb72:		; preds = %bb49, %bb46
	br i1 false, label %bb49, label %bb77
bb77:		; preds = %bb72
	%mrv.gep3 = getelementptr %struct.Demand* %mrv, i32 0, i32 0		; <double*> [#uses=1]
	%mrv.ld4 = load double* %mrv.gep3		; <double> [#uses=1]
	%mrv.gep5 = getelementptr %struct.Demand* %mrv, i32 0, i32 1		; <double*> [#uses=1]
	%mrv.ld6 = load double* %mrv.gep5		; <double> [#uses=1]
	ret double %mrv.ld4, double %mrv.ld6
}

define %struct.Demand @Compute_Leaf1(%struct.leaf* %l, double %pi_R, double %pi_I) nounwind  {
entry:
	%mrv = alloca %struct.Demand		; <%struct.Demand*> [#uses=2]
	%tmp10 = load double* @P, align 8		; <double> [#uses=1]
	%tmp11 = fcmp olt double %tmp10, 0.000000e+00		; <i1> [#uses=1]
	br i1 %tmp11, label %bb, label %bb13
bb:		; preds = %entry
	br label %bb13
bb13:		; preds = %bb, %entry
	%mrv.gep = getelementptr %struct.Demand* %mrv, i32 0, i32 0		; <double*> [#uses=1]
	%mrv.ld = load double* %mrv.gep		; <double> [#uses=1]
	%mrv.gep1 = getelementptr %struct.Demand* %mrv, i32 0, i32 1		; <double*> [#uses=1]
	%mrv.ld2 = load double* %mrv.gep1		; <double> [#uses=1]
	ret double %mrv.ld, double %mrv.ld2
}
