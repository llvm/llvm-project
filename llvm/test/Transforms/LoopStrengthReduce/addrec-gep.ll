; RUN: opt -opaque-pointers=0 < %s -loop-reduce -S | FileCheck %s
; CHECK: bb1:
; CHECK: load double, double* [[IV:%[^,]+]]
; CHECK: store double {{.*}}, double* [[IV]]
; CHECK: getelementptr double, double*
; CHECK-NOT: cast
; CHECK: br {{.*}} label %bb1

; This test tests several things. The load and store should use the
; same address instead of having it computed twice, and SCEVExpander should
; be able to reconstruct the full getelementptr, despite it having a few
; obstacles set in its way.
; We only check that the inner loop (bb1-bb2) is "reduced" because LSR
; currently only operates on inner loops.

target datalayout = "e-p:64:64:64-n32:64"

define void @foo(i64 %n, i64 %m, i64 %o, i64 %q, double* nocapture %p) nounwind {
entry:
	%tmp = icmp sgt i64 %n, 0		; <i1> [#uses=1]
	br i1 %tmp, label %bb.nph3, label %return

bb.nph:		; preds = %bb2.preheader
	%tmp1 = mul i64 %tmp16, %i.02		; <i64> [#uses=1]
	%tmp2 = mul i64 %tmp19, %i.02		; <i64> [#uses=1]
	br label %bb1

bb1:		; preds = %bb2, %bb.nph
	%j.01 = phi i64 [ %tmp9, %bb2 ], [ 0, %bb.nph ]		; <i64> [#uses=3]
	%tmp3 = add i64 %j.01, %tmp1		; <i64> [#uses=1]
	%tmp4 = add i64 %j.01, %tmp2		; <i64> [#uses=1]
        %z0 = add i64 %tmp3, 5203
	%tmp5 = getelementptr double, double* %p, i64 %z0		; <double*> [#uses=1]
	%tmp6 = load double, double* %tmp5, align 8		; <double> [#uses=1]
	%tmp7 = fdiv double %tmp6, 2.100000e+00		; <double> [#uses=1]
        %z1 = add i64 %tmp4, 5203
	%tmp8 = getelementptr double, double* %p, i64 %z1		; <double*> [#uses=1]
	store double %tmp7, double* %tmp8, align 8
	%tmp9 = add i64 %j.01, 1		; <i64> [#uses=2]
	br label %bb2

bb2:		; preds = %bb1
	%tmp10 = icmp slt i64 %tmp9, %m		; <i1> [#uses=1]
	br i1 %tmp10, label %bb1, label %bb2.bb3_crit_edge

bb2.bb3_crit_edge:		; preds = %bb2
	br label %bb3

bb3:		; preds = %bb2.preheader, %bb2.bb3_crit_edge
	%tmp11 = add i64 %i.02, 1		; <i64> [#uses=2]
	br label %bb4

bb4:		; preds = %bb3
	%tmp12 = icmp slt i64 %tmp11, %n		; <i1> [#uses=1]
	br i1 %tmp12, label %bb2.preheader, label %bb4.return_crit_edge

bb4.return_crit_edge:		; preds = %bb4
	br label %bb4.return_crit_edge.split

bb4.return_crit_edge.split:		; preds = %bb.nph3, %bb4.return_crit_edge
	br label %return

bb.nph3:		; preds = %entry
	%tmp13 = icmp sgt i64 %m, 0		; <i1> [#uses=1]
	%tmp14 = mul i64 %n, 37		; <i64> [#uses=1]
	%tmp15 = mul i64 %tmp14, %o		; <i64> [#uses=1]
	%tmp16 = mul i64 %tmp15, %q		; <i64> [#uses=1]
	%tmp17 = mul i64 %n, 37		; <i64> [#uses=1]
	%tmp18 = mul i64 %tmp17, %o		; <i64> [#uses=1]
	%tmp19 = mul i64 %tmp18, %q		; <i64> [#uses=1]
	br i1 %tmp13, label %bb.nph3.split, label %bb4.return_crit_edge.split

bb.nph3.split:		; preds = %bb.nph3
	br label %bb2.preheader

bb2.preheader:		; preds = %bb.nph3.split, %bb4
	%i.02 = phi i64 [ %tmp11, %bb4 ], [ 0, %bb.nph3.split ]		; <i64> [#uses=3]
	br i1 true, label %bb.nph, label %bb3

return:		; preds = %bb4.return_crit_edge.split, %entry
	ret void
}
