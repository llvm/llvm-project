; RUN: llvm-as < %s | opt -predsimplify -disable-output

define i32 @test_wp_B_slice(i32 %select_method) {
entry:
	br label %bb309
cond_true114:		; preds = %bb309
	%tmp130 = icmp slt i32 0, 128		; <i1> [#uses=1]
	%min = select i1 %tmp130, i32 0, i32 127		; <i32> [#uses=2]
	%tmp143 = load i32* null		; <i32> [#uses=0]
	br i1 false, label %bb303, label %bb314
cond_true166:		; preds = %bb303
	ret i32 0
cond_false200:		; preds = %bb303
	%tmp205 = sdiv i32 %min, 2		; <i32> [#uses=1]
	%iftmp.380.0.p = select i1 false, i32 0, i32 %tmp205		; <i32> [#uses=0]
	ret i32 0
bb303:		; preds = %cond_true114
	%tmp165 = icmp eq i32 %min, 0		; <i1> [#uses=1]
	br i1 %tmp165, label %cond_true166, label %cond_false200
bb309:		; preds = %entry
	br i1 false, label %cond_true114, label %bb314
bb314:		; preds = %bb309, %cond_true114
	ret i32 0
}

