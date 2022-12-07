; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output
; PR584
@g_38098584 = external global i32		; <i32*> [#uses=1]
@g_60187400 = external global i32		; <i32*> [#uses=1]
@g_59182229 = external global i32		; <i32*> [#uses=2]

define i32 @_Z13func_26556482h(i8 %l_88173906) {
entry:
	%tmp.1 = bitcast i8 %l_88173906 to i8		; <i8> [#uses=2]
	%tmp.3 = icmp eq i8 %l_88173906, 0		; <i1> [#uses=1]
	br i1 %tmp.3, label %else.0, label %then.0
then.0:		; preds = %entry
	%tmp.5 = icmp eq i8 %l_88173906, 0		; <i1> [#uses=1]
	br i1 %tmp.5, label %else.1, label %then.1
then.1:		; preds = %then.0
	br label %return
else.1:		; preds = %then.0
	br label %loopentry.0
loopentry.0:		; preds = %no_exit.0, %else.1
	%i.0.1 = phi i32 [ 0, %else.1 ], [ %inc.0, %no_exit.0 ]		; <i32> [#uses=2]
	%tmp.9 = icmp sgt i32 %i.0.1, 99		; <i1> [#uses=1]
	br i1 %tmp.9, label %endif.0, label %no_exit.0
no_exit.0:		; preds = %loopentry.0
	%inc.0 = add i32 %i.0.1, 1		; <i32> [#uses=1]
	br label %loopentry.0
else.0:		; preds = %entry
	%tmp.12 = sext i8 %tmp.1 to i32		; <i32> [#uses=1]
	br label %return
endif.0:		; preds = %loopentry.0
	%tmp.14 = sext i8 %tmp.1 to i32		; <i32> [#uses=1]
	%tmp.16 = zext i8 %l_88173906 to i32		; <i32> [#uses=1]
	%tmp.17 = icmp sgt i32 %tmp.14, %tmp.16		; <i1> [#uses=1]
	%tmp.19 = load i32, i32* @g_59182229		; <i32> [#uses=2]
	br i1 %tmp.17, label %cond_true, label %cond_false
cond_true:		; preds = %endif.0
	%tmp.20 = icmp ne i32 %tmp.19, 1		; <i1> [#uses=1]
	br label %cond_continue
cond_false:		; preds = %endif.0
	%tmp.22 = icmp ne i32 %tmp.19, 0		; <i1> [#uses=1]
	br label %cond_continue
cond_continue:		; preds = %cond_false, %cond_true
	%mem_tmp.0 = phi i1 [ %tmp.20, %cond_true ], [ %tmp.22, %cond_false ]		; <i1> [#uses=1]
	br i1 %mem_tmp.0, label %then.2, label %else.2
then.2:		; preds = %cond_continue
	%tmp.25 = zext i8 %l_88173906 to i32		; <i32> [#uses=1]
	br label %return
else.2:		; preds = %cond_continue
	br label %loopentry.1
loopentry.1:		; preds = %endif.3, %else.2
	%i.1.1 = phi i32 [ 0, %else.2 ], [ %inc.3, %endif.3 ]		; <i32> [#uses=2]
	%i.3.2 = phi i32 [ undef, %else.2 ], [ %i.3.0, %endif.3 ]		; <i32> [#uses=2]
	%l_88173906_addr.1 = phi i8 [ %l_88173906, %else.2 ], [ %l_88173906_addr.0, %endif.3 ]		; <i8> [#uses=3]
	%tmp.29 = icmp sgt i32 %i.1.1, 99		; <i1> [#uses=1]
	br i1 %tmp.29, label %endif.2, label %no_exit.1
no_exit.1:		; preds = %loopentry.1
	%tmp.30 = load i32, i32* @g_38098584		; <i32> [#uses=1]
	%tmp.31 = icmp eq i32 %tmp.30, 0		; <i1> [#uses=1]
	br i1 %tmp.31, label %else.3, label %then.3
then.3:		; preds = %no_exit.1
	br label %endif.3
else.3:		; preds = %no_exit.1
	br i1 false, label %else.4, label %then.4
then.4:		; preds = %else.3
	br label %endif.3
else.4:		; preds = %else.3
	br i1 false, label %else.5, label %then.5
then.5:		; preds = %else.4
	store i32 -1004318825, i32* @g_59182229
	br label %return
else.5:		; preds = %else.4
	br label %loopentry.3
loopentry.3:		; preds = %then.7, %else.5
	%i.3.3 = phi i32 [ 0, %else.5 ], [ %inc.2, %then.7 ]		; <i32> [#uses=3]
	%tmp.55 = icmp sgt i32 %i.3.3, 99		; <i1> [#uses=1]
	br i1 %tmp.55, label %endif.3, label %no_exit.3
no_exit.3:		; preds = %loopentry.3
	%tmp.57 = icmp eq i8 %l_88173906_addr.1, 0		; <i1> [#uses=1]
	br i1 %tmp.57, label %else.7, label %then.7
then.7:		; preds = %no_exit.3
	store i32 16239, i32* @g_60187400
	%inc.2 = add i32 %i.3.3, 1		; <i32> [#uses=1]
	br label %loopentry.3
else.7:		; preds = %no_exit.3
	br label %return
endif.3:		; preds = %loopentry.3, %then.4, %then.3
	%i.3.0 = phi i32 [ %i.3.2, %then.3 ], [ %i.3.2, %then.4 ], [ %i.3.3, %loopentry.3 ]		; <i32> [#uses=1]
	%l_88173906_addr.0 = phi i8 [ 100, %then.3 ], [ %l_88173906_addr.1, %then.4 ], [ %l_88173906_addr.1, %loopentry.3 ]		; <i8> [#uses=1]
	%inc.3 = add i32 %i.1.1, 1		; <i32> [#uses=1]
	br label %loopentry.1
endif.2:		; preds = %loopentry.1
	br label %return
return:		; preds = %endif.2, %else.7, %then.5, %then.2, %else.0, %then.1
	%result.0 = phi i32 [ 1624650671, %then.1 ], [ %tmp.25, %then.2 ], [ 3379, %then.5 ], [ 52410, %else.7 ], [ -1526438411, %endif.2 ], [ %tmp.12, %else.0 ]		; <i32> [#uses=1]
	ret i32 %result.0
}
