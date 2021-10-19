; RUN: opt < %s -inline -instcombine -jump-threading -licm -simple-loop-unswitch -instcombine -indvars -loop-deletion -gvn -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -verify -disable-output

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.BF_BitstreamElement = type { i32, i16 }
	%struct.BF_BitstreamPart = type { i32, %struct.BF_BitstreamElement* }
	%struct.BF_PartHolder = type { i32, %struct.BF_BitstreamPart* }
	%struct.Bit_stream_struc = type { i8*, i32, %struct.FILE*, i8*, i32, i32, i32, i32 }
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.III_scalefac_t = type { [22 x i32], [13 x [3 x i32]] }
	%struct.III_side_info_t = type { i32, i32, i32, [2 x [4 x i32]], [2 x %struct.anon] }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.anon = type { [2 x %struct.gr_info_ss] }
	%struct.gr_info = type { i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32*, [4 x i32] }
	%struct.gr_info_ss = type { %struct.gr_info }
	%struct.lame_global_flags = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, float, float, float, float, i32, i32, i32, i32, i32, i32, i32, i32 }
@scaleFactorsPH = external global [2 x [2 x %struct.BF_PartHolder*]]		; <[2 x [2 x %struct.BF_PartHolder*]]*> [#uses=1]
@slen1_tab = external constant [16 x i32]		; <[16 x i32]*> [#uses=1]

declare %struct.BF_PartHolder* @BF_addElement(%struct.BF_PartHolder*, %struct.BF_BitstreamElement*) nounwind

define %struct.BF_PartHolder* @BF_addEntry(%struct.BF_PartHolder* %thePH, i32 %value, i32 %length) nounwind  {
entry:
	%myElement = alloca %struct.BF_BitstreamElement		; <%struct.BF_BitstreamElement*> [#uses=2]
	%tmp1 = getelementptr %struct.BF_BitstreamElement, %struct.BF_BitstreamElement* %myElement, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %value, i32* %tmp1, align 8
	%tmp7 = icmp eq i32 %length, 0		; <i1> [#uses=1]
	br i1 %tmp7, label %bb13, label %bb

bb:		; preds = %entry
	%tmp10 = call %struct.BF_PartHolder* @BF_addElement( %struct.BF_PartHolder* %thePH, %struct.BF_BitstreamElement* %myElement ) nounwind 		; <%struct.BF_PartHolder*> [#uses=1]
	ret %struct.BF_PartHolder* %tmp10

bb13:		; preds = %entry
	ret %struct.BF_PartHolder* %thePH
}

define void @III_format_bitstream(%struct.lame_global_flags* %gfp, i32 %bitsPerFrame, [2 x [576 x i32]]* %l3_enc, %struct.III_side_info_t* %l3_side, [2 x %struct.III_scalefac_t]* %scalefac, %struct.Bit_stream_struc* %in_bs) nounwind  {
entry:
	call fastcc void @encodeMainData( %struct.lame_global_flags* %gfp, [2 x [576 x i32]]* %l3_enc, %struct.III_side_info_t* %l3_side, [2 x %struct.III_scalefac_t]* %scalefac ) nounwind
	unreachable
}

define internal fastcc void @encodeMainData(%struct.lame_global_flags* %gfp, [2 x [576 x i32]]* %l3_enc, %struct.III_side_info_t* %si, [2 x %struct.III_scalefac_t]* %scalefac) nounwind  {
entry:
	%tmp69 = getelementptr %struct.lame_global_flags, %struct.lame_global_flags* %gfp, i32 0, i32 43		; <i32*> [#uses=1]
	%tmp70 = load i32, i32* %tmp69, align 4		; <i32> [#uses=1]
	%tmp71 = icmp eq i32 %tmp70, 1		; <i1> [#uses=1]
	br i1 %tmp71, label %bb352, label %bb498

bb113:		; preds = %bb132
	%tmp123 = getelementptr [2 x %struct.III_scalefac_t], [2 x %struct.III_scalefac_t]* %scalefac, i32 0, i32 0, i32 1, i32 %sfb.0, i32 %window.0		; <i32*> [#uses=1]
	%tmp124 = load i32, i32* %tmp123, align 4		; <i32> [#uses=1]
	%tmp126 = load %struct.BF_PartHolder*, %struct.BF_PartHolder** %tmp80, align 4		; <%struct.BF_PartHolder*> [#uses=1]
	%tmp128 = call %struct.BF_PartHolder* @BF_addEntry( %struct.BF_PartHolder* %tmp126, i32 %tmp124, i32 %tmp93 ) nounwind 		; <%struct.BF_PartHolder*> [#uses=1]
	store %struct.BF_PartHolder* %tmp128, %struct.BF_PartHolder** %tmp80, align 4
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
	%tmp160 = load %struct.BF_PartHolder*, %struct.BF_PartHolder** %tmp80, align 4		; <%struct.BF_PartHolder*> [#uses=1]
	%tmp162 = call %struct.BF_PartHolder* @BF_addEntry( %struct.BF_PartHolder* %tmp160, i32 0, i32 0 ) nounwind 		; <%struct.BF_PartHolder*> [#uses=0]
	unreachable

bb174:		; preds = %bb140
	%tmp176 = icmp slt i32 6, 12		; <i1> [#uses=1]
	br i1 %tmp176, label %bb166, label %bb341

bb341:		; preds = %bb352, %bb174
	%tmp80 = getelementptr [2 x [2 x %struct.BF_PartHolder*]], [2 x [2 x %struct.BF_PartHolder*]]* @scaleFactorsPH, i32 0, i32 0, i32 0		; <%struct.BF_PartHolder**> [#uses=3]
	%tmp92 = getelementptr [16 x i32], [16 x i32]* @slen1_tab, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp93 = load i32, i32* %tmp92, align 4		; <i32> [#uses=1]
	br label %bb140

bb352:		; preds = %entry
	%tmp354 = icmp slt i32 0, 2		; <i1> [#uses=1]
	br i1 %tmp354, label %bb341, label %return

bb498:		; preds = %entry
	ret void

return:		; preds = %bb352
	ret void
}

define void @getframebits(%struct.lame_global_flags* %gfp, i32* %bitsPerFrame, i32* %mean_bits) nounwind  {
entry:
	unreachable
}

define i32 @lame_encode_buffer(%struct.lame_global_flags* %gfp, i16* %buffer_l, i16* %buffer_r, i32 %nsamples, i8* %mp3buf, i32 %mp3buf_size) nounwind  {
entry:
	unreachable
}
