; RUN: llvm-as < %s | opt -sccp -dce -simplifycfg | llvm-dis | \
; RUN:   not grep br

define i32 @test(i32 %param) {
entry:
	%tmp.1 = icmp ne i32 %param, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %endif.0, label %else
else:		; preds = %entry
	br label %endif.0
endif.0:		; preds = %else, %entry
	%a.0 = phi i32 [ 2, %else ], [ 3, %entry ]		; <i32> [#uses=1]
	%b.0 = phi i32 [ 3, %else ], [ 2, %entry ]		; <i32> [#uses=1]
	%tmp.5 = add i32 %a.0, %b.0		; <i32> [#uses=1]
	%tmp.7 = icmp ne i32 %tmp.5, 5		; <i1> [#uses=1]
	br i1 %tmp.7, label %UnifiedReturnBlock, label %endif.1
endif.1:		; preds = %endif.0
	ret i32 0
UnifiedReturnBlock:		; preds = %endif.0
	ret i32 2
}

