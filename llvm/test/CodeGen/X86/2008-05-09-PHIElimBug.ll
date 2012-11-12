; RUN: llc < %s -march=x86

	%struct.V = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x i32>, float*, float*, float*, float*, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, i32, i32, i32, i32, i32, i32, i32, i32 }

define fastcc void @t() nounwind  {
entry:
	br i1 false, label %bb23816.preheader, label %bb23821

bb23816.preheader:		; preds = %entry
	%tmp23735 = and i32 0, 2		; <i32> [#uses=0]
	br label %bb23830

bb23821:		; preds = %entry
	br i1 false, label %bb23830, label %bb23827

bb23827:		; preds = %bb23821
	%tmp23829 = getelementptr %struct.V* null, i32 0, i32 42		; <i32*> [#uses=0]
	br label %bb23830

bb23830:		; preds = %bb23827, %bb23821, %bb23816.preheader
	%scaledInDst.2.reg2mem.5 = phi i8 [ undef, %bb23827 ], [ undef, %bb23821 ], [ undef, %bb23816.preheader ]		; <i8> [#uses=1]
	%toBool35047 = icmp eq i8 %scaledInDst.2.reg2mem.5, 0		; <i1> [#uses=1]
	%bothcond39107 = or i1 %toBool35047, false		; <i1> [#uses=0]
	unreachable
}
