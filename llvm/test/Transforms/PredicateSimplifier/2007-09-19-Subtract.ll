; RUN: llvm-as < %s | opt -predsimplify | llvm-dis | grep unreachable | count 2
; PR1683

@.str = internal constant [13 x i8] c"c36174a.adb\00\00"		; <[13 x i8]*> [#uses=1]

define void @_ada_c36174a() {
entry:
	%tmp3 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%tmp4 = invoke i32 @report__ident_int( i32 6 )
			to label %invcont unwind label %entry.lpad_crit_edge		; <i32> [#uses=7]

entry.lpad_crit_edge:		; preds = %entry
	br label %lpad

invcont:		; preds = %entry
	%tmp6 = icmp slt i32 %tmp4, 1		; <i1> [#uses=1]
	br i1 %tmp6, label %bb, label %bb9

bb:		; preds = %invcont
	invoke void @__gnat_rcheck_07( i8* getelementptr ([13 x i8]* @.str, i32 0, i32 0), i32 10 )
			to label %invcont8 unwind label %bb.lpad_crit_edge

bb.lpad_crit_edge:		; preds = %bb
	br label %lpad

invcont8:		; preds = %bb
	unreachable

bb9:		; preds = %invcont
	%tmp51 = add i32 %tmp4, 6		; <i32> [#uses=2]
	%tmp56 = icmp sgt i32 %tmp4, %tmp51		; <i1> [#uses=1]
	br i1 %tmp56, label %bb9.bb76_crit_edge, label %bb9.bb61_crit_edge

bb9.bb61_crit_edge:		; preds = %bb9
	br label %bb61

bb9.bb76_crit_edge:		; preds = %bb9
	br label %bb76

bb61:		; preds = %bb73, %bb9.bb61_crit_edge
	%J4b.0 = phi i32 [ %tmp75, %bb73 ], [ %tmp4, %bb9.bb61_crit_edge ]		; <i32> [#uses=2]
	%tmp70 = icmp eq i32 %tmp51, %J4b.0		; <i1> [#uses=1]
	br i1 %tmp70, label %bb61.bb76_crit_edge, label %bb73

bb61.bb76_crit_edge:		; preds = %bb61
	br label %bb76

bb73:		; preds = %bb61
	%tmp75 = add i32 %J4b.0, 1		; <i32> [#uses=1]
	br label %bb61

bb76:		; preds = %bb61.bb76_crit_edge, %bb9.bb76_crit_edge
	%tmp78 = icmp ne i32 %tmp4, 6		; <i1> [#uses=1]
	%tmp81 = add i32 %tmp4, 6		; <i32> [#uses=1]
	%tmp8182 = sext i32 %tmp81 to i64		; <i64> [#uses=1]
	%tmp8384 = sext i32 %tmp4 to i64		; <i64> [#uses=1]
	%tmp85 = sub i64 %tmp8182, %tmp8384		; <i64> [#uses=1]
	%tmp86 = icmp ne i64 %tmp85, 6		; <i1> [#uses=1]
	%tmp90 = or i1 %tmp78, %tmp86		; <i1> [#uses=1]
	br i1 %tmp90, label %bb93, label %bb76.bb99_crit_edge

bb76.bb99_crit_edge:		; preds = %bb76
	br label %bb99

bb93:		; preds = %bb76
	invoke void @abort( )
			to label %bb93.bb99_crit_edge unwind label %bb93.lpad_crit_edge

bb93.lpad_crit_edge:		; preds = %bb93
	br label %lpad

bb93.bb99_crit_edge:		; preds = %bb93
	br label %bb99

bb99:		; preds = %bb93.bb99_crit_edge, %bb76.bb99_crit_edge
	ret void

lpad:		; preds = %bb93.lpad_crit_edge, %bb.lpad_crit_edge, %entry.lpad_crit_edge
	%eh_ptr = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select102 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32 0 )		; <i32> [#uses=0]
	call void @llvm.stackrestore( i8* %tmp3 )
	call i32 (...)* @_Unwind_Resume( i8* %eh_ptr )		; <i32>:0 [#uses=0]
	unreachable
}

declare i8* @llvm.stacksave()

declare i32 @report__ident_int(i32)

declare void @__gnat_rcheck_07(i8*, i32)

declare void @abort()

declare i8* @llvm.eh.exception()

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...)

declare i32 @__gnat_eh_personality(...)

declare i32 @_Unwind_Resume(...)

declare void @llvm.stackrestore(i8*)
