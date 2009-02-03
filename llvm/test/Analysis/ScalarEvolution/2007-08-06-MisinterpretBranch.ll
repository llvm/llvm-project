; RUN: llvm-as < %s | opt -indvars -adce -simplifycfg | llvm-dis | grep "icmp s"
; PR1598

define i32 @f(i32 %a, i32 %b, i32 %x, i32 %y) {
entry:
	%tmp3 = icmp eq i32 %a, %b		; <i1> [#uses=1]
	br i1 %tmp3, label %return, label %bb

bb:		; preds = %bb, %entry
	%x_addr.0 = phi i32 [ %tmp6, %bb ], [ %x, %entry ]		; <i32> [#uses=1]
	%tmp6 = add i32 %x_addr.0, 1		; <i32> [#uses=3]
	%tmp9 = icmp slt i32 %tmp6, %y		; <i1> [#uses=1]
	br i1 %tmp9, label %bb, label %return

return:		; preds = %bb, %entry
	%x_addr.1 = phi i32 [ %x, %entry ], [ %tmp6, %bb ]		; <i32> [#uses=1]
	ret i32 %x_addr.1
}
