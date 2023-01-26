; RUN: opt < %s -passes=licm -disable-output
; PR908
; END.

	%struct.alloc_chain = type { ptr, ptr }
	%struct.oggpack_buffer = type { i32, i32, ptr, ptr, i32 }
	%struct.vorbis_block = type { ptr, %struct.oggpack_buffer, i32, i32, i32, i32, i32, i32, i64, i64, ptr, ptr, i32, i32, i32, ptr, i32, i32, i32, i32, ptr }
	%struct.vorbis_dsp_state = type { i32, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, ptr }
	%struct.vorbis_info = type { i32, i32, i32, i32, i32, i32, i32, ptr }

define fastcc void @_01forward() {
entry:
	br i1 false, label %bb222.preheader, label %bb241
cond_true67:		; preds = %cond_true87
	br label %cond_next80
cond_next80:		; preds = %cond_true87, %cond_true67
	br label %bb83
bb83.preheader:		; preds = %cond_true226
	br i1 false, label %bb83.us.preheader, label %bb83.preheader1
bb83.us.preheader:		; preds = %bb83.preheader
	br label %bb83.us
bb83.us:		; preds = %cond_next80.us, %bb83.us.preheader
	br i1 false, label %cond_true87.us, label %cond_next92.loopexit2
cond_next80.us:		; preds = %bb59.loopexit.us, %cond_true67.us
	br label %bb83.us
cond_true67.us:		; preds = %bb59.loopexit.us
	br label %cond_next80.us
cond_next.us:		; preds = %cond_true56.us, %cond_true38.us
	br i1 false, label %cond_true56.us, label %bb59.loopexit.us
cond_true38.us:		; preds = %cond_true56.us
	br label %cond_next.us
cond_true56.us:		; preds = %cond_true87.us, %cond_next.us
	br i1 false, label %cond_true38.us, label %cond_next.us
cond_true87.us:		; preds = %bb83.us
	br label %cond_true56.us
bb59.loopexit.us:		; preds = %cond_next.us
	br i1 false, label %cond_true67.us, label %cond_next80.us
bb83.preheader1:		; preds = %bb83.preheader
	br label %bb83
bb83:		; preds = %bb83.preheader1, %cond_next80
	br i1 false, label %cond_next92.loopexit, label %cond_true87
cond_true87:		; preds = %bb83
	br i1 false, label %cond_true67, label %cond_next80
cond_next92.loopexit:		; preds = %bb83
	br label %cond_next92
cond_next92.loopexit2:		; preds = %bb83.us
	br label %cond_next92
cond_next92:		; preds = %cond_true226, %cond_next92.loopexit2, %cond_next92.loopexit
	br i1 false, label %cond_true218.loopexit, label %bb222
cond_true139:		; preds = %cond_true202
	br i1 false, label %cond_next195, label %cond_true155
cond_true155:		; preds = %cond_true139
	br i1 false, label %cond_true249.i.preheader, label %_encodepart.exit
cond_true.i:		; preds = %cond_true115.i
	br i1 false, label %bb60.i.preheader, label %cond_next97.i
bb60.i.preheader:		; preds = %cond_true.i
	br label %bb60.i
bb60.i:		; preds = %cond_true63.i, %bb60.i.preheader
	br i1 false, label %cond_true63.i, label %cond_next97.i.loopexit
cond_true63.i:		; preds = %bb60.i
	br i1 false, label %bb60.i, label %cond_next97.i.loopexit
bb86.i.preheader:		; preds = %cond_true115.i
	br label %bb86.i
bb86.i:		; preds = %cond_true93.i, %bb86.i.preheader
	br i1 false, label %cond_true93.i, label %cond_next97.i.loopexit3
cond_true93.i:		; preds = %bb86.i
	br i1 false, label %cond_next97.i.loopexit3, label %bb86.i
cond_next97.i.loopexit:		; preds = %cond_true63.i, %bb60.i
	br label %cond_next97.i
cond_next97.i.loopexit3:		; preds = %cond_true93.i, %bb86.i
	br label %cond_next97.i
cond_next97.i:		; preds = %cond_next97.i.loopexit3, %cond_next97.i.loopexit, %cond_true.i
	br i1 false, label %bb118.i.loopexit, label %cond_true115.i
cond_true115.i.preheader:		; preds = %cond_true249.i
	br label %cond_true115.i
cond_true115.i:		; preds = %cond_true115.i.preheader, %cond_next97.i
	br i1 false, label %cond_true.i, label %bb86.i.preheader
bb118.i.loopexit:		; preds = %cond_next97.i
	br label %bb118.i
bb118.i:		; preds = %cond_true249.i, %bb118.i.loopexit
	br i1 false, label %cond_next204.i, label %cond_true128.i
cond_true128.i:		; preds = %bb118.i
	br i1 false, label %cond_true199.i.preheader, label %cond_next204.i
cond_true199.i.preheader:		; preds = %cond_true128.i
	br label %cond_true199.i
cond_true199.i.us:		; No predecessors!
	br i1 false, label %cond_true167.i.us, label %cond_next187.i.us
cond_next187.i.us:		; preds = %bb170.i.loopexit.us, %bb170.i.us.cond_next187.i.us_crit_edge, %cond_true199.i.us
	unreachable
bb170.i.us.cond_next187.i.us_crit_edge:		; preds = %bb170.i.loopexit.us
	br label %cond_next187.i.us
cond_true167.i.us:		; preds = %cond_true167.i.us, %cond_true199.i.us
	br i1 false, label %cond_true167.i.us, label %bb170.i.loopexit.us
bb170.i.loopexit.us:		; preds = %cond_true167.i.us
	br i1 false, label %cond_next187.i.us, label %bb170.i.us.cond_next187.i.us_crit_edge
cond_true199.i:		; preds = %cond_true199.i, %cond_true199.i.preheader
	br i1 false, label %cond_next204.i.loopexit, label %cond_true199.i
cond_next204.i.loopexit:		; preds = %cond_true199.i
	br label %cond_next204.i
cond_next204.i:		; preds = %cond_next204.i.loopexit, %cond_true128.i, %bb118.i
	br label %bb233.i
cond_true230.i:		; No predecessors!
	%exitcond155 = icmp eq i32 0, %tmp16.i		; <i1> [#uses=0]
	unreachable
bb233.i:		; preds = %cond_next204.i
	br i1 false, label %_encodepart.exit.loopexit, label %cond_true249.i
cond_true249.i.preheader:		; preds = %cond_true155
	br label %cond_true249.i
cond_true249.i:		; preds = %cond_true249.i.preheader, %bb233.i
	%tmp16.i = bitcast i32 0 to i32		; <i32> [#uses=1]
	br i1 false, label %cond_true115.i.preheader, label %bb118.i
_encodepart.exit.loopexit:		; preds = %bb233.i
	br label %_encodepart.exit
_encodepart.exit:		; preds = %_encodepart.exit.loopexit, %cond_true155
	br label %cond_next195
cond_next195:		; preds = %cond_true202, %_encodepart.exit, %cond_true139
	br i1 false, label %bb205.loopexit, label %cond_true202
cond_true202.preheader:		; preds = %cond_true218
	br label %cond_true202
cond_true202:		; preds = %cond_true202.preheader, %cond_next195
	br i1 false, label %cond_next195, label %cond_true139
bb205.loopexit:		; preds = %cond_next195
	br label %bb205
bb205:		; preds = %cond_true218, %bb205.loopexit
	br i1 false, label %cond_true218, label %bb222.outer105.loopexit
cond_true218.loopexit:		; preds = %cond_next92
	br label %cond_true218
cond_true218:		; preds = %cond_true218.loopexit, %bb205
	br i1 false, label %cond_true202.preheader, label %bb205
bb222.preheader:		; preds = %entry
	br label %bb222.outer
bb222.outer:		; preds = %bb229, %bb222.preheader
	br label %bb222.outer105
bb222.outer105.loopexit:		; preds = %bb205
	br label %bb222.outer105
bb222.outer105:		; preds = %bb222.outer105.loopexit, %bb222.outer
	br label %bb222
bb222:		; preds = %bb222.outer105, %cond_next92
	br i1 false, label %cond_true226, label %bb229
cond_true226:		; preds = %bb222
	br i1 false, label %bb83.preheader, label %cond_next92
bb229:		; preds = %bb222
	br i1 false, label %bb222.outer, label %bb241.loopexit
bb241.loopexit:		; preds = %bb229
	br label %bb241
bb241:		; preds = %bb241.loopexit, %entry
	ret void
}
