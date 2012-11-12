; RUN: llc < %s -o - -march=x86 -mattr=+mmx | FileCheck %s
; There are no MMX instructions here.  We use add+adcl for the adds.

define <1 x i64> @unsigned_add3(<1 x i64>* %a, <1 x i64>* %b, i32 %count) nounwind {
entry:
	%tmp2942 = icmp eq i32 %count, 0		; <i1> [#uses=1]
	br i1 %tmp2942, label %bb31, label %bb26

bb26:		; preds = %bb26, %entry

; CHECK:  addl
; CHECK:  adcl

	%i.037.0 = phi i32 [ 0, %entry ], [ %tmp25, %bb26 ]		; <i32> [#uses=3]
	%sum.035.0 = phi <1 x i64> [ zeroinitializer, %entry ], [ %tmp22, %bb26 ]		; <<1 x i64>> [#uses=1]
	%tmp13 = getelementptr <1 x i64>* %b, i32 %i.037.0		; <<1 x i64>*> [#uses=1]
	%tmp14 = load <1 x i64>* %tmp13		; <<1 x i64>> [#uses=1]
	%tmp18 = getelementptr <1 x i64>* %a, i32 %i.037.0		; <<1 x i64>*> [#uses=1]
	%tmp19 = load <1 x i64>* %tmp18		; <<1 x i64>> [#uses=1]
	%tmp21 = add <1 x i64> %tmp19, %tmp14		; <<1 x i64>> [#uses=1]
	%tmp22 = add <1 x i64> %tmp21, %sum.035.0		; <<1 x i64>> [#uses=2]
	%tmp25 = add i32 %i.037.0, 1		; <i32> [#uses=2]
	%tmp29 = icmp ult i32 %tmp25, %count		; <i1> [#uses=1]
	br i1 %tmp29, label %bb26, label %bb31

bb31:		; preds = %bb26, %entry
	%sum.035.1 = phi <1 x i64> [ zeroinitializer, %entry ], [ %tmp22, %bb26 ]		; <<1 x i64>> [#uses=1]
	ret <1 x i64> %sum.035.1
}


; This is the original test converted to use MMX intrinsics.

define <1 x i64> @unsigned_add3a(x86_mmx* %a, x86_mmx* %b, i32 %count) nounwind {
entry:
        %tmp2943 = bitcast <1 x i64><i64 0> to x86_mmx
	%tmp2942 = icmp eq i32 %count, 0		; <i1> [#uses=1]
	br i1 %tmp2942, label %bb31, label %bb26

bb26:		; preds = %bb26, %entry

; CHECK:  movq	({{.*}},8), %mm
; CHECK:  paddq	({{.*}},8), %mm
; CHECK:  paddq	%mm{{[0-7]}}, %mm

	%i.037.0 = phi i32 [ 0, %entry ], [ %tmp25, %bb26 ]		; <i32> [#uses=3]
	%sum.035.0 = phi x86_mmx [ %tmp2943, %entry ], [ %tmp22, %bb26 ]		; <x86_mmx> [#uses=1]
	%tmp13 = getelementptr x86_mmx* %b, i32 %i.037.0		; <x86_mmx*> [#uses=1]
	%tmp14 = load x86_mmx* %tmp13		; <x86_mmx> [#uses=1]
	%tmp18 = getelementptr x86_mmx* %a, i32 %i.037.0		; <x86_mmx*> [#uses=1]
	%tmp19 = load x86_mmx* %tmp18		; <x86_mmx> [#uses=1]
	%tmp21 = call x86_mmx @llvm.x86.mmx.padd.q (x86_mmx %tmp19, x86_mmx %tmp14)		; <x86_mmx> [#uses=1]
	%tmp22 = call x86_mmx @llvm.x86.mmx.padd.q (x86_mmx %tmp21, x86_mmx %sum.035.0)		; <x86_mmx> [#uses=2]
	%tmp25 = add i32 %i.037.0, 1		; <i32> [#uses=2]
	%tmp29 = icmp ult i32 %tmp25, %count		; <i1> [#uses=1]
	br i1 %tmp29, label %bb26, label %bb31

bb31:		; preds = %bb26, %entry
	%sum.035.1 = phi x86_mmx [ %tmp2943, %entry ], [ %tmp22, %bb26 ]		; <x86_mmx> [#uses=1]
        %t = bitcast x86_mmx %sum.035.1 to <1 x i64>
	ret <1 x i64> %t
}

declare x86_mmx @llvm.x86.mmx.padd.q(x86_mmx, x86_mmx)
