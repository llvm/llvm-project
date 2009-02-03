; RUN: llvm-as < %s | opt -tailduplicate -instcombine -simplifycfg -licm -disable-output
target datalayout = "e-p:32:32"
@yy_base = external global [787 x i16]		; <[787 x i16]*> [#uses=1]
@yy_state_ptr = external global i32*		; <i32**> [#uses=3]
@yy_state_buf = external global [16386 x i32]		; <[16386 x i32]*> [#uses=1]
@yy_lp = external global i32		; <i32*> [#uses=1]

define i32 @_yylex() {
	br label %loopentry.0
loopentry.0:		; preds = %else.26, %0
	store i32* getelementptr ([16386 x i32]* @yy_state_buf, i64 0, i64 0), i32** @yy_state_ptr
	%tmp.35 = load i32** @yy_state_ptr		; <i32*> [#uses=2]
	%inc.0 = getelementptr i32* %tmp.35, i64 1		; <i32*> [#uses=1]
	store i32* %inc.0, i32** @yy_state_ptr
	%tmp.36 = load i32* null		; <i32> [#uses=1]
	store i32 %tmp.36, i32* %tmp.35
	br label %loopexit.2
loopexit.2:		; preds = %else.26, %loopexit.2, %loopentry.0
	store i8* null, i8** null
	%tmp.91 = load i32* null		; <i32> [#uses=1]
	%tmp.92 = sext i32 %tmp.91 to i64		; <i64> [#uses=1]
	%tmp.93 = getelementptr [787 x i16]* @yy_base, i64 0, i64 %tmp.92		; <i16*> [#uses=1]
	%tmp.94 = load i16* %tmp.93		; <i16> [#uses=1]
	%tmp.95 = icmp ne i16 %tmp.94, 4394		; <i1> [#uses=1]
	br i1 %tmp.95, label %loopexit.2, label %yy_find_action
yy_find_action:		; preds = %else.26, %loopexit.2
	br label %loopentry.3
loopentry.3:		; preds = %then.9, %shortcirc_done.0, %yy_find_action
	%tmp.105 = load i32* @yy_lp		; <i32> [#uses=1]
	%tmp.106 = icmp ne i32 %tmp.105, 0		; <i1> [#uses=1]
	br i1 %tmp.106, label %shortcirc_next.0, label %shortcirc_done.0
shortcirc_next.0:		; preds = %loopentry.3
	%tmp.114 = load i16* null		; <i16> [#uses=1]
	%tmp.115 = sext i16 %tmp.114 to i32		; <i32> [#uses=1]
	%tmp.116 = icmp slt i32 0, %tmp.115		; <i1> [#uses=1]
	br label %shortcirc_done.0
shortcirc_done.0:		; preds = %shortcirc_next.0, %loopentry.3
	%shortcirc_val.0 = phi i1 [ false, %loopentry.3 ], [ %tmp.116, %shortcirc_next.0 ]		; <i1> [#uses=1]
	br i1 %shortcirc_val.0, label %else.0, label %loopentry.3
else.0:		; preds = %shortcirc_done.0
	%tmp.144 = load i32* null		; <i32> [#uses=1]
	%tmp.145 = and i32 %tmp.144, 8192		; <i32> [#uses=1]
	%tmp.146 = icmp ne i32 %tmp.145, 0		; <i1> [#uses=1]
	br i1 %tmp.146, label %then.9, label %else.26
then.9:		; preds = %else.0
	br label %loopentry.3
else.26:		; preds = %else.0
	switch i32 0, label %loopentry.0 [
		 i32 2, label %yy_find_action
		 i32 0, label %loopexit.2
	]
}
