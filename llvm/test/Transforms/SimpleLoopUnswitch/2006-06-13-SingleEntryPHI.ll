; RUN: opt < %s -simple-loop-unswitch -disable-output
; RUN: opt < %s -simple-loop-unswitch -enable-mssa-loop-dependency=true -verify-memoryssa -disable-output

	%struct.BLEND_MAP = type { i16, i16, i16, i32, %struct.BLEND_MAP_ENTRY* }
	%struct.BLEND_MAP_ENTRY = type { float, i8, { [5 x float], [4 x i8] } }
	%struct.TPATTERN = type { i16, i16, i16, i32, float, float, float, %struct.WARP*, %struct.TPATTERN*, %struct.BLEND_MAP*, { %struct.anon, [4 x i8] } }
	%struct.TURB = type { i16, %struct.WARP*, [3 x double], i32, float, float }
	%struct.WARP = type { i16, %struct.WARP* }
	%struct.anon = type { float, [3 x double] }

define void @Parse_Pattern() {
entry:
	br label %bb1096.outer20
bb671:		; preds = %cond_true1099
	br label %bb1096.outer23
bb1096.outer20.loopexit:		; preds = %cond_true1099
	%Local_Turb.0.ph24.lcssa = phi %struct.TURB* [ %Local_Turb.0.ph24, %cond_true1099 ]		; <%struct.TURB*> [#uses=1]
	br label %bb1096.outer20
bb1096.outer20:		; preds = %bb1096.outer20.loopexit, %entry
	%Local_Turb.0.ph22 = phi %struct.TURB* [ undef, %entry ], [ %Local_Turb.0.ph24.lcssa, %bb1096.outer20.loopexit ]		; <%struct.TURB*> [#uses=1]
	%tmp1098 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br label %bb1096.outer23
bb1096.outer23:		; preds = %bb1096.outer20, %bb671
	%Local_Turb.0.ph24 = phi %struct.TURB* [ %Local_Turb.0.ph22, %bb1096.outer20 ], [ null, %bb671 ]		; <%struct.TURB*> [#uses=2]
	br label %bb1096
bb1096:		; preds = %cond_true1099, %bb1096.outer23
	br i1 %tmp1098, label %cond_true1099, label %bb1102
cond_true1099:		; preds = %bb1096
	switch i32 0, label %bb1096.outer20.loopexit [
		 i32 161, label %bb671
		 i32 359, label %bb1096
	]
bb1102:		; preds = %bb1096
	%Local_Turb.0.ph24.lcssa1 = phi %struct.TURB* [ %Local_Turb.0.ph24, %bb1096 ]		; <%struct.TURB*> [#uses=0]
	ret void
}
