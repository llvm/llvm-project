; RUN: opt < %s -passes='cgscc(inline),function(instcombine,jump-threading,loop-mssa(licm),simple-loop-unswitch,instcombine,indvars,loop-deletion,gvn,simplifycfg),verify' -simplifycfg-require-and-preserve-domtree=1 -disable-output

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.BF_BitstreamElement = type { i32, i16 }
	%struct.BF_BitstreamPart = type { i32, ptr }
	%struct.BF_PartHolder = type { i32, ptr }
	%struct.Bit_stream_struc = type { ptr, i32, ptr, ptr, i32, i32, i32, i32 }
	%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.III_scalefac_t = type { [22 x i32], [13 x [3 x i32]] }
	%struct.III_side_info_t = type { i32, i32, i32, [2 x [4 x i32]], [2 x %struct.anon] }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
	%struct.anon = type { [2 x %struct.gr_info_ss] }
	%struct.gr_info = type { i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, [4 x i32] }
	%struct.gr_info_ss = type { %struct.gr_info }
	%struct.lame_global_flags = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, float, float, float, float, i32, i32, i32, i32, i32, i32, i32, i32 }
@scaleFactorsPH = external global [2 x [2 x ptr]]		; <ptr> [#uses=1]
@slen1_tab = external constant [16 x i32]		; <ptr> [#uses=1]

declare ptr @BF_addElement(ptr, ptr) nounwind

define ptr @BF_addEntry(ptr %thePH, i32 %value, i32 %length) nounwind  {
entry:
	%myElement = alloca %struct.BF_BitstreamElement		; <ptr> [#uses=2]
	%tmp1 = getelementptr %struct.BF_BitstreamElement, ptr %myElement, i32 0, i32 0		; <ptr> [#uses=1]
	store i32 %value, ptr %tmp1, align 8
	%tmp7 = icmp eq i32 %length, 0		; <i1> [#uses=1]
	br i1 %tmp7, label %bb13, label %bb

bb:		; preds = %entry
	%tmp10 = call ptr @BF_addElement( ptr %thePH, ptr %myElement ) nounwind 		; <ptr> [#uses=1]
	ret ptr %tmp10

bb13:		; preds = %entry
	ret ptr %thePH
}

define void @III_format_bitstream(ptr %gfp, i32 %bitsPerFrame, ptr %l3_enc, ptr %l3_side, ptr %scalefac, ptr %in_bs) nounwind  {
entry:
	call fastcc void @encodeMainData( ptr %gfp, ptr %l3_enc, ptr %l3_side, ptr %scalefac ) nounwind
	unreachable
}

define internal fastcc void @encodeMainData(ptr %gfp, ptr %l3_enc, ptr %si, ptr %scalefac) nounwind  {
entry:
	%tmp69 = getelementptr %struct.lame_global_flags, ptr %gfp, i32 0, i32 43		; <ptr> [#uses=1]
	%tmp70 = load i32, ptr %tmp69, align 4		; <i32> [#uses=1]
	%tmp71 = icmp eq i32 %tmp70, 1		; <i1> [#uses=1]
	br i1 %tmp71, label %bb352, label %bb498

bb113:		; preds = %bb132
	%tmp123 = getelementptr [2 x %struct.III_scalefac_t], ptr %scalefac, i32 0, i32 0, i32 1, i32 %sfb.0, i32 %window.0		; <ptr> [#uses=1]
	%tmp124 = load i32, ptr %tmp123, align 4		; <i32> [#uses=1]
	%tmp126 = load ptr, ptr %tmp80, align 4		; <ptr> [#uses=1]
	%tmp128 = call ptr @BF_addEntry( ptr %tmp126, i32 %tmp124, i32 %tmp93 ) nounwind 		; <ptr> [#uses=1]
	store ptr %tmp128, ptr %tmp80, align 4
	%tmp131 = add i32 %window.0, 1		; <i32> [#uses=1]
	br label %bb132

bb132:		; preds = %bb140, %bb113
	%window.0 = phi i32 [ %tmp131, %bb113 ], [ 0, %bb140 ]		; <i32> [#uses=3]
	%tmp134 = icmp slt i32 %window.0, 3		; <i1> [#uses=1]
	br i1 %tmp134, label %bb113, label %bb137

bb137:		; preds = %bb132
	%tmp139 = add i32 %sfb.0, 1		; <i32> [#uses=1]
	br label %bb140

bb140:		; preds = %bb341, %bb137
	%sfb.0 = phi i32 [ %tmp139, %bb137 ], [ 0, %bb341 ]		; <i32> [#uses=3]
	%tmp142 = icmp slt i32 %sfb.0, 6		; <i1> [#uses=1]
	br i1 %tmp142, label %bb132, label %bb174

bb166:		; preds = %bb174
	%tmp160 = load ptr, ptr %tmp80, align 4		; <ptr> [#uses=1]
	%tmp162 = call ptr @BF_addEntry( ptr %tmp160, i32 0, i32 0 ) nounwind 		; <ptr> [#uses=0]
	unreachable

bb174:		; preds = %bb140
	%tmp176 = icmp slt i32 6, 12		; <i1> [#uses=1]
	br i1 %tmp176, label %bb166, label %bb341

bb341:		; preds = %bb352, %bb174
	%tmp80 = getelementptr [2 x [2 x ptr]], ptr @scaleFactorsPH, i32 0, i32 0, i32 0		; <ptr> [#uses=3]
	%tmp92 = getelementptr [16 x i32], ptr @slen1_tab, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp93 = load i32, ptr %tmp92, align 4		; <i32> [#uses=1]
	br label %bb140

bb352:		; preds = %entry
	%tmp354 = icmp slt i32 0, 2		; <i1> [#uses=1]
	br i1 %tmp354, label %bb341, label %return

bb498:		; preds = %entry
	ret void

return:		; preds = %bb352
	ret void
}

define void @getframebits(ptr %gfp, ptr %bitsPerFrame, ptr %mean_bits) nounwind  {
entry:
	unreachable
}

define i32 @lame_encode_buffer(ptr %gfp, ptr %buffer_l, ptr %buffer_r, i32 %nsamples, ptr %mp3buf, i32 %mp3buf_size) nounwind  {
entry:
	unreachable
}
