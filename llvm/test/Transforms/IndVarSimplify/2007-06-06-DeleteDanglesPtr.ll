; RUN: opt < %s -passes=indvars -disable-output
; PR1487

	%struct.AVClass = type { ptr, ptr, ptr }
	%struct.AVCodec = type { ptr, i32, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr }
	%struct.AVCodecContext = type { ptr, i32, i32, i32, i32, i32, ptr, i32, %struct.AVCodecTag, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, float, float, i32, i32, i32, i32, float, i32, i32, i32, ptr, ptr, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, [32 x i8], i32, i32, i32, i32, i32, i32, i32, float, i32, ptr, ptr, i32, i32, i32, i32, ptr, ptr, float, float, i32, ptr, i32, ptr, i32, i32, i32, float, float, float, float, i32, float, float, float, float, float, i32, i32, i32, ptr, i32, i32, i32, i32, %struct.AVCodecTag, ptr, i32, i32, [4 x i64], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, i32, i32, i32, ptr, i32, ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64 }
	%struct.AVCodecTag = type { i32, i32 }
	%struct.AVFrame = type { [4 x ptr], [4 x i32], [4 x ptr], i32, i32, i64, i32, i32, i32, i32, i32, ptr, i32, ptr, [2 x ptr], ptr, i8, ptr, [4 x i64], i32, i32, i32, i32, i32, ptr, i32, i32, ptr, [2 x ptr] }
	%struct.AVOption = type { ptr, ptr, i32, i32, double, double, double, i32, ptr }
	%struct.AVPaletteControl = type { i32, [256 x i32] }
	%struct.AVPanScan = type { i32, i32, i32, [3 x [2 x i16]] }
	%struct.RcOverride = type { i32, i32, i32, float }

define i32 @smc_decode_frame(ptr %avctx, ptr %data, ptr %data_size, ptr %buf, i32 %buf_size) {
entry:
	br i1 false, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	ret i32 -1

cond_next:		; preds = %entry
	br i1 false, label %bb.outer5.split.split.split.us, label %cond_true194.split

bb.outer5.split.split.split.us:		; preds = %cond_next
	br i1 false, label %cond_next188.us503.us, label %bb.us481

bb275.us493.us:		; preds = %cond_next188.us503.us, %cond_next188.us503.us
	ret i32 0

cond_next188.us503.us:		; preds = %bb.outer5.split.split.split.us
	switch i32 0, label %bb1401 [
		 i32 0, label %cond_next202.bb215_crit_edge.split
		 i32 16, label %bb215
		 i32 32, label %bb275.us493.us
		 i32 48, label %bb275.us493.us
		 i32 64, label %cond_next202.bb417_crit_edge.split
		 i32 80, label %bb417
		 i32 96, label %cond_next202.bb615_crit_edge.split
		 i32 112, label %bb615
		 i32 128, label %cond_next202.bb716_crit_edge.split
		 i32 144, label %bb716
		 i32 160, label %cond_next202.bb882_crit_edge.split
		 i32 176, label %bb882
		 i32 192, label %cond_next202.bb1062_crit_edge.split
		 i32 208, label %bb1062
		 i32 224, label %bb1326.us.outer.outer
	]

bb.us481:		; preds = %bb.outer5.split.split.split.us
	ret i32 0

cond_true194.split:		; preds = %cond_next
	ret i32 %buf_size

cond_next202.bb1062_crit_edge.split:		; preds = %cond_next188.us503.us
	ret i32 0

cond_next202.bb882_crit_edge.split:		; preds = %cond_next188.us503.us
	ret i32 0

cond_next202.bb716_crit_edge.split:		; preds = %cond_next188.us503.us
	ret i32 0

cond_next202.bb615_crit_edge.split:		; preds = %cond_next188.us503.us
	ret i32 0

cond_next202.bb417_crit_edge.split:		; preds = %cond_next188.us503.us
	ret i32 0

cond_next202.bb215_crit_edge.split:		; preds = %cond_next188.us503.us
	ret i32 0

bb215:		; preds = %cond_next188.us503.us
	ret i32 0

bb417:		; preds = %cond_next188.us503.us
	ret i32 0

bb615:		; preds = %cond_next188.us503.us
	ret i32 0

bb716:		; preds = %cond_next188.us503.us
	ret i32 0

bb882:		; preds = %cond_next188.us503.us
	ret i32 0

bb1062:		; preds = %cond_next188.us503.us
	ret i32 0

bb1326.us:		; preds = %bb1326.us.outer.outer, %bb1347.loopexit.us, %bb1326.us
	%pixel_y.162036.us.ph = phi i32 [ %tmp1352.us, %bb1347.loopexit.us ], [ 0, %bb1326.us.outer.outer ], [ %pixel_y.162036.us.ph, %bb1326.us ]		; <i32> [#uses=2]
	%stream_ptr.142038.us.ph = phi i32 [ %tmp1339.us, %bb1347.loopexit.us ], [ %stream_ptr.142038.us.ph.ph, %bb1326.us.outer.outer ], [ %stream_ptr.142038.us.ph, %bb1326.us ]		; <i32> [#uses=2]
	%pixel_x.232031.us = phi i32 [ %tmp1341.us, %bb1326.us ], [ 0, %bb1326.us.outer.outer ], [ 0, %bb1347.loopexit.us ]		; <i32> [#uses=3]
	%block_ptr.222030.us = add i32 0, %pixel_x.232031.us		; <i32> [#uses=1]
	%stream_ptr.132032.us = add i32 %pixel_x.232031.us, %stream_ptr.142038.us.ph		; <i32> [#uses=1]
	%tmp1341.us = add i32 %pixel_x.232031.us, 1		; <i32> [#uses=2]
	%tmp1344.us = icmp slt i32 %tmp1341.us, 4		; <i1> [#uses=1]
	br i1 %tmp1344.us, label %bb1326.us, label %bb1347.loopexit.us

bb1347.loopexit.us:		; preds = %bb1326.us
	%tmp1339.us = add i32 %stream_ptr.132032.us, 1		; <i32> [#uses=2]
	%tmp1337.us = add i32 %block_ptr.222030.us, 1		; <i32> [#uses=0]
	%tmp1352.us = add i32 %pixel_y.162036.us.ph, 1		; <i32> [#uses=2]
	%tmp1355.us = icmp slt i32 %tmp1352.us, 4		; <i1> [#uses=1]
	br i1 %tmp1355.us, label %bb1326.us, label %bb1358

bb1358:		; preds = %bb1347.loopexit.us
	br label %bb1326.us.outer.outer

bb1326.us.outer.outer:		; preds = %bb1358, %cond_next188.us503.us
	%stream_ptr.142038.us.ph.ph = phi i32 [ %tmp1339.us, %bb1358 ], [ 0, %cond_next188.us503.us ]		; <i32> [#uses=1]
	br label %bb1326.us

bb1401:		; preds = %cond_next188.us503.us
	ret i32 0
}
