; RUN: llc < %s -mtriple=i386-apple-darwin
; PR2757

@g_3 = external global i32		; <i32*> [#uses=1]

define i32 @func_125(i32 %p_126, i32 %p_128, i32 %p_129) nounwind {
entry:
	%tmp2.i = load i32* @g_3		; <i32> [#uses=2]
	%conv = trunc i32 %tmp2.i to i16		; <i16> [#uses=3]
	br label %forcond1.preheader.i.i7

forcond1.preheader.i.i7:		; preds = %forinc6.i.i25, %entry
	%p_86.addr.06.i.i4 = phi i32 [ 0, %entry ], [ %sub.i.i.i23, %forinc6.i.i25 ]		; <i32> [#uses=1]
	%p_87.addr.15.i.i5 = phi i32 [ 0, %entry ], [ %p_87.addr.0.lcssa.i.i21, %forinc6.i.i25 ]		; <i32> [#uses=2]
	br i1 false, label %forinc6.i.i25, label %forinc.i.i11

forinc.i.i11:		; preds = %forcond1.backedge.i.i20, %forcond1.preheader.i.i7
	%p_87.addr.02.i.i8 = phi i32 [ %p_87.addr.15.i.i5, %forcond1.preheader.i.i7 ], [ %p_87.addr.0.be.i.i18, %forcond1.backedge.i.i20 ]		; <i32> [#uses=1]
	%conv.i.i9 = trunc i32 %p_87.addr.02.i.i8 to i8		; <i8> [#uses=1]
	br i1 false, label %land_rhs3.i.i.i14, label %lor_rhs.i.i.i17

land_rhs3.i.i.i14:		; preds = %forinc.i.i11
	br i1 false, label %forcond1.backedge.i.i20, label %lor_rhs.i.i.i17

lor_rhs.i.i.i17:		; preds = %land_rhs3.i.i.i14, %forinc.i.i11
	%conv29.i.i.i15 = sext i8 %conv.i.i9 to i32		; <i32> [#uses=1]
	%add.i.i.i16 = add i32 %conv29.i.i.i15, 1		; <i32> [#uses=1]
	br label %forcond1.backedge.i.i20

forcond1.backedge.i.i20:		; preds = %lor_rhs.i.i.i17, %land_rhs3.i.i.i14
	%p_87.addr.0.be.i.i18 = phi i32 [ %add.i.i.i16, %lor_rhs.i.i.i17 ], [ 0, %land_rhs3.i.i.i14 ]		; <i32> [#uses=3]
	%tobool3.i.i19 = icmp eq i32 %p_87.addr.0.be.i.i18, 0		; <i1> [#uses=1]
	br i1 %tobool3.i.i19, label %forinc6.i.i25, label %forinc.i.i11

forinc6.i.i25:		; preds = %forcond1.backedge.i.i20, %forcond1.preheader.i.i7
	%p_87.addr.0.lcssa.i.i21 = phi i32 [ %p_87.addr.15.i.i5, %forcond1.preheader.i.i7 ], [ %p_87.addr.0.be.i.i18, %forcond1.backedge.i.i20 ]		; <i32> [#uses=1]
	%conv.i.i.i22 = and i32 %p_86.addr.06.i.i4, 255		; <i32> [#uses=1]
	%sub.i.i.i23 = add i32 %conv.i.i.i22, -1		; <i32> [#uses=2]
	%phitmp.i.i24 = icmp eq i32 %sub.i.i.i23, 0		; <i1> [#uses=1]
	br i1 %phitmp.i.i24, label %func_106.exit27, label %forcond1.preheader.i.i7

func_106.exit27:		; preds = %forinc6.i.i25
	%cmp = icmp ne i32 %tmp2.i, 1		; <i1> [#uses=3]
	%cmp.ext = zext i1 %cmp to i32		; <i32> [#uses=1]
	br i1 %cmp, label %safe_mod_int16_t_s_s.exit, label %lor_rhs.i

lor_rhs.i:		; preds = %func_106.exit27
	%tobool.i = xor i1 %cmp, true		; <i1> [#uses=1]
	%or.cond.i = or i1 false, %tobool.i		; <i1> [#uses=1]
	br i1 %or.cond.i, label %ifend.i, label %safe_mod_int16_t_s_s.exit

ifend.i:		; preds = %lor_rhs.i
	%conv6.i = sext i16 %conv to i32		; <i32> [#uses=1]
	%rem.i = urem i32 %conv6.i, %cmp.ext		; <i32> [#uses=1]
	%conv8.i = trunc i32 %rem.i to i16		; <i16> [#uses=1]
	br label %safe_mod_int16_t_s_s.exit

safe_mod_int16_t_s_s.exit:		; preds = %ifend.i, %lor_rhs.i, %func_106.exit27
	%call31 = phi i16 [ %conv8.i, %ifend.i ], [ %conv, %func_106.exit27 ], [ %conv, %lor_rhs.i ]		; <i16> [#uses=1]
	%conv4 = sext i16 %call31 to i32		; <i32> [#uses=1]
	%call5 = tail call i32 (...)* @func_104( i32 %conv4 )		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @func_104(...)
