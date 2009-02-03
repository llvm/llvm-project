; RUN: llvm-as < %s | llc -o - -march=x86 -mattr=+mmx | grep paddq | count 2
; RUN: llvm-as < %s | llc -o - -march=x86 -mattr=+mmx | grep movq | count 2

define <1 x i64> @unsigned_add3(<1 x i64>* %a, <1 x i64>* %b, i32 %count) {
entry:
	%tmp2942 = icmp eq i32 %count, 0		; <i1> [#uses=1]
	br i1 %tmp2942, label %bb31, label %bb26

bb26:		; preds = %bb26, %entry
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
