; RUN: llvm-as < %s | llc 
; PR1424

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "arm-linux-gnueabi"
	%struct.AVClass = type { i8*, i8* (i8*)*, %struct.AVOption* }
	%struct.AVCodec = type { i8*, i32, i32, i32, i32 (%struct.AVCodecContext*)*, i32 (%struct.AVCodecContext*, i8*, i32, i8*)*, i32 (%struct.AVCodecContext*)*, i32 (%struct.AVCodecContext*, i8*, i32*, i8*, i32)*, i32, %struct.AVCodec*, void (%struct.AVCodecContext*)*, %struct.AVRational*, i32* }
	%struct.AVCodecContext = type { %struct.AVClass*, i32, i32, i32, i32, i32, i8*, i32, %struct.AVRational, i32, i32, i32, i32, i32, void (%struct.AVCodecContext*, %struct.AVFrame*, i32*, i32, i32, i32)*, i32, i32, i32, i32, i32, i32, i32, float, float, i32, i32, i32, i32, float, i32, i32, i32, %struct.AVCodec*, i8*, i32, i32, void (%struct.AVCodecContext*, i8*, i32, i32)*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, [32 x i8], i32, i32, i32, i32, i32, i32, i32, float, i32, i32 (%struct.AVCodecContext*, %struct.AVFrame*)*, void (%struct.AVCodecContext*, %struct.AVFrame*)*, i32, i32, i32, i32, i8*, i8*, float, float, i32, %struct.RcOverride*, i32, i8*, i32, i32, i32, float, float, float, float, i32, float, float, float, float, float, i32, i32, i32, i32*, i32, i32, i32, i32, %struct.AVRational, %struct.AVFrame*, i32, i32, [4 x i64], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 (%struct.AVCodecContext*, i32*)*, i32, i32, i32, i32, i32, i32, i8*, i32, i32, i32, i32, i32, i32, i16*, i16*, i32, i32, i32, i32, %struct.AVPaletteControl*, i32, i32 (%struct.AVCodecContext*, %struct.AVFrame*)*, i32, i32, i32, i32, i32, i32, i32, i32 (%struct.AVCodecContext*, i32 (%struct.AVCodecContext*, i8*)*, i8**, i32*, i32)*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64 }
	%struct.AVEvalExpr = type opaque
	%struct.AVFrame = type { [4 x i8*], [4 x i32], [4 x i8*], i32, i32, i64, i32, i32, i32, i32, i32, i8*, i32, i8*, [2 x [2 x i16]*], i32*, i8, i8*, [4 x i64], i32, i32, i32, i32, i32, %struct.AVPanScan*, i32, i32, i16*, [2 x i8*] }
	%struct.AVOption = type opaque
	%struct.AVPaletteControl = type { i32, [256 x i32] }
	%struct.AVPanScan = type { i32, i32, i32, [3 x [2 x i16]] }
	%struct.AVRational = type { i32, i32 }
	%struct.BlockNode = type { i16, i16, i8, [3 x i8], i8, i8 }
	%struct.DSPContext = type { void (i16*, i8*, i32)*, void (i16*, i8*, i8*, i32)*, void (i16*, i8*, i32)*, void (i16*, i8*, i32)*, void (i16*, i8*, i32)*, void (i8*, i16*, i32)*, void (i8*, i16*, i32)*, i32 (i16*)*, void (i8*, i8*, i32, i32, i32, i32, i32)*, void (i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)*, void (i16*)*, i32 (i8*, i32)*, i32 (i8*, i32)*, [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], i32 (i8*, i16*, i32)*, [4 x [4 x void (i8*, i8*, i32, i32)*]], [4 x [4 x void (i8*, i8*, i32, i32)*]], [4 x [4 x void (i8*, i8*, i32, i32)*]], [4 x [4 x void (i8*, i8*, i32, i32)*]], [2 x void (i8*, i8*, i8*, i32, i32)*], [11 x void (i8*, i8*, i32, i32, i32)*], [11 x void (i8*, i8*, i32, i32, i32)*], [2 x [16 x void (i8*, i8*, i32)*]], [2 x [16 x void (i8*, i8*, i32)*]], [2 x [16 x void (i8*, i8*, i32)*]], [2 x [16 x void (i8*, i8*, i32)*]], [8 x void (i8*, i8*, i32)*], [3 x void (i8*, i8*, i32, i32, i32, i32)*], [3 x void (i8*, i8*, i32, i32, i32, i32)*], [3 x void (i8*, i8*, i32, i32, i32, i32)*], [4 x [16 x void (i8*, i8*, i32)*]], [4 x [16 x void (i8*, i8*, i32)*]], [4 x [16 x void (i8*, i8*, i32)*]], [4 x [16 x void (i8*, i8*, i32)*]], [10 x void (i8*, i32, i32, i32, i32)*], [10 x void (i8*, i8*, i32, i32, i32, i32, i32)*], [2 x [16 x void (i8*, i8*, i32)*]], [2 x [16 x void (i8*, i8*, i32)*]], void (i8*, i32, i32, i32, i32, i32, i32)*, void (i8*, i32, i32, i32, i32, i32, i32)*, void (i8*, i32, i32, i32, i32, i32, i32)*, void (i8*, i32, i32, i32, i32, i32, i32)*, void (i8*, i16*, i32)*, [2 x [4 x i32 (i8*, i8*, i8*, i32, i32)*]], void (i8*, i8*, i32)*, void (i8*, i8*, i8*, i32)*, void (i8*, i8*, i8*, i32, i32*, i32*)*, void (i32*, i32*, i32)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32)*, void (i8*, i32, i32, i32)*, void ([4 x [4 x i16]]*, i8*, [40 x i8]*, [40 x [2 x i16]]*, i32, i32, i32, i32, i32)*, void (i8*, i32, i32)*, void (i8*, i32, i32)*, void (i8*, i32)*, void (float*, float*, i32)*, void (float*, float*, i32)*, void (float*, float*, float*, i32)*, void (float*, float*, float*, float*, i32, i32, i32)*, void (i16*, float*, i32)*, void (i16*)*, void (i16*)*, void (i16*)*, void (i8*, i32, i16*)*, void (i8*, i32, i16*)*, [64 x i8], i32, i32 (i16*, i16*, i16*, i32)*, void (i16*, i16*, i32)*, void (i8*, i16*, i32)*, void (i8*, i16*, i32)*, void (i8*, i16*, i32)*, void (i8*, i16*, i32)*, void ([4 x i16]*)*, void (i32*, i32*, i32*, i32*, i32*, i32*, i32)*, void (i32*, i32)*, void (i8*, i32, i8**, i32, i32, i32, i32, i32, %struct.slice_buffer*, i32, i8*)*, void (i8*, i32, i32)*, [4 x void (i8*, i32, i8*, i32, i32, i32)*], void (i16*)*, void (i16*, i32)*, void (i16*, i32)*, void (i16*, i32)*, void (i8*, i32)*, void (i8*, i32)*, [16 x void (i8*, i8*, i32, i32)*] }
	%struct.FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct.GetBitContext = type { i8*, i8*, i32*, i32, i32, i32, i32 }
	%struct.MJpegContext = type opaque
	%struct.MotionEstContext = type { %struct.AVCodecContext*, i32, [4 x [2 x i32]], [4 x [2 x i32]], i8*, i8*, [2 x i8*], i8*, i32, i32*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [4 x [4 x i8*]], [4 x [4 x i8*]], i32, i32, i32, i32, i32, [4 x void (i8*, i8*, i32, i32)*]*, [4 x void (i8*, i8*, i32, i32)*]*, [16 x void (i8*, i8*, i32)*]*, [16 x void (i8*, i8*, i32)*]*, [4097 x i8]*, i8*, i32 (%struct.MpegEncContext*, i32*, i32*, i32, i32, i32, i32, i32)* }
	%struct.MpegEncContext = type { %struct.AVCodecContext*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.PutBitContext, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.Picture*, %struct.Picture**, %struct.Picture**, i32, i32, [8 x %struct.MpegEncContext*], %struct.Picture, %struct.Picture, %struct.Picture, %struct.Picture, %struct.Picture*, %struct.Picture*, %struct.Picture*, [3 x i8*], [3 x i32], i16*, [3 x i16*], [20 x i16], i32, i32, i8*, i8*, i8*, i8*, i8*, [16 x i16]*, [3 x [16 x i16]*], i32, i8*, i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, [5 x i32], i32, i32, i32, i32, %struct.DSPContext, i32, i32, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x [2 x [2 x i16]*]], [2 x [2 x [2 x [2 x i16]*]]], [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x [2 x [2 x i16]*]], [2 x [2 x [2 x [2 x i16]*]]], [2 x i8*], [2 x [2 x i8*]], i32, i32, i32, [2 x [4 x [2 x i32]]], [2 x [2 x i32]], [2 x [2 x [2 x i32]]], i8*, [2 x [64 x i16]], %struct.MotionEstContext, i32, i32, i32, i32, i32, i32, i16*, [6 x i32], [6 x i32], [3 x i8*], i32*, [64 x i16], [64 x i16], [64 x i16], [64 x i16], i32, i32, i32, i32, i32, i8*, i8*, i8*, i8*, i8*, i8*, [8 x i32], [64 x i32]*, [64 x i32]*, [2 x [64 x i16]]*, [2 x [64 x i16]]*, [12 x i32], %struct.ScanTable, %struct.ScanTable, %struct.ScanTable, %struct.ScanTable, [64 x i32]*, [2 x i32], [64 x i16]*, i8*, i64, i64, i32, i32, %struct.RateControlContext, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i32, i32, %struct.GetBitContext, i32, i32, i32, %struct.ParseContext, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i16, i16, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x [2 x i32]], [2 x [2 x i32]], [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.PutBitContext, %struct.PutBitContext, i32, i32, i32, i32, i32, i32, i8*, i32, i32, i32, i32, i32, [3 x i32], %struct.MJpegContext*, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x [65 x [65 x [2 x i32]]]]*, i32, i32, %struct.GetBitContext, i32, i32, i32, i8*, i32, [2 x [2 x i32]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, i32, i32, i32, i8*, i32, [12 x i16*], [64 x i16]*, [8 x [64 x i16]]*, i32 (%struct.MpegEncContext*, [64 x i16]*)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, i32 (%struct.MpegEncContext*, i16*, i32, i32, i32*)*, i32 (%struct.MpegEncContext*, i16*, i32, i32, i32*)*, void (%struct.MpegEncContext*, i16*)* }
	%struct.ParseContext = type { i8*, i32, i32, i32, i32, i32, i32, i32 }
	%struct.Picture = type { [4 x i8*], [4 x i32], [4 x i8*], i32, i32, i64, i32, i32, i32, i32, i32, i8*, i32, i8*, [2 x [2 x i16]*], i32*, i8, i8*, [4 x i64], i32, i32, i32, i32, i32, %struct.AVPanScan*, i32, i32, i16*, [2 x i8*], [3 x i8*], [2 x [2 x i16]*], i32*, [2 x i32], i32, i32, i32, i32, [2 x [16 x i32]], [2 x i32], i32, i32, i16*, i16*, i8*, i32*, i32 }
	%struct.Plane = type { i32, i32, [8 x [4 x %struct.SubBand]] }
	%struct.Predictor = type { double, double, double }
	%struct.PutBitContext = type { i32, i32, i8*, i8*, i8* }
	%struct.RangeCoder = type { i32, i32, i32, i32, [256 x i8], [256 x i8], i8*, i8*, i8* }
	%struct.RateControlContext = type { %struct.FILE*, i32, %struct.RateControlEntry*, double, [5 x %struct.Predictor], double, double, double, double, double, [5 x double], i32, i32, [5 x i64], [5 x i64], [5 x i64], [5 x i64], [5 x i32], i32, i8*, float, i32, %struct.AVEvalExpr* }
	%struct.RateControlEntry = type { i32, float, i32, i32, i32, i32, i32, i64, i32, float, i32, i32, i32, i32, i32, i32 }
	%struct.RcOverride = type { i32, i32, i32, float }
	%struct.ScanTable = type { i8*, [64 x i8], [64 x i8] }
	%struct.SnowContext = type { %struct.AVCodecContext*, %struct.RangeCoder, %struct.DSPContext, %struct.AVFrame, %struct.AVFrame, %struct.AVFrame, [8 x %struct.AVFrame], %struct.AVFrame, [32 x i8], [4224 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [8 x [2 x i16]*], [8 x i32*], i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [4 x %struct.Plane], %struct.BlockNode*, [1024 x i32], i32, %struct.slice_buffer, %struct.MpegEncContext }
	%struct.SubBand = type { i32, i32, i32, i32, i32, i32*, i32, i32, i32, %struct.x_and_coeff*, %struct.SubBand*, [519 x [32 x i8]] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, i32 }
	%struct.slice_buffer = type { i32**, i32**, i32, i32, i32, i32, i32* }
	%struct.x_and_coeff = type { i16, i16 }

define fastcc void @iterative_me(%struct.SnowContext* %s) {
entry:
	%state = alloca [4224 x i8], align 8		; <[4224 x i8]*> [#uses=0]
	%best_rd4233 = alloca i32, align 4		; <i32*> [#uses=0]
	%tmp21 = getelementptr %struct.SnowContext* %s, i32 0, i32 36		; <i32*> [#uses=2]
	br label %bb4198

bb79:		; preds = %bb4189.preheader
	br i1 false, label %cond_next239, label %cond_true

cond_true:		; preds = %bb79
	ret void

cond_next239:		; preds = %bb79
	%tmp286 = alloca i8, i32 0		; <i8*> [#uses=0]
	ret void

bb4198:		; preds = %bb4189.preheader, %entry
	br i1 false, label %bb4189.preheader, label %bb4204

bb4189.preheader:		; preds = %bb4198
	br i1 false, label %bb79, label %bb4198

bb4204:		; preds = %bb4198
	br i1 false, label %bb4221, label %cond_next4213

cond_next4213:		; preds = %bb4204
	ret void

bb4221:		; preds = %bb4204
	br i1 false, label %bb5242.preheader, label %UnifiedReturnBlock

bb5242.preheader:		; preds = %bb4221
	br label %bb5242

bb4231:		; preds = %bb5233
	%tmp4254.sum = add i32 0, 1		; <i32> [#uses=2]
	br i1 false, label %bb4559, label %cond_next4622

bb4559:		; preds = %bb4231
	ret void

cond_next4622:		; preds = %bb4231
	%tmp4637 = load i16* null		; <i16> [#uses=1]
	%tmp46374638 = sext i16 %tmp4637 to i32		; <i32> [#uses=1]
	%tmp4642 = load i16* null		; <i16> [#uses=1]
	%tmp46424643 = sext i16 %tmp4642 to i32		; <i32> [#uses=1]
	%tmp4648 = load i16* null		; <i16> [#uses=1]
	%tmp46484649 = sext i16 %tmp4648 to i32		; <i32> [#uses=1]
	%tmp4653 = getelementptr %struct.BlockNode* null, i32 %tmp4254.sum, i32 0		; <i16*> [#uses=1]
	%tmp4654 = load i16* %tmp4653		; <i16> [#uses=1]
	%tmp46544655 = sext i16 %tmp4654 to i32		; <i32> [#uses=1]
	%tmp4644 = add i32 %tmp46374638, 2		; <i32> [#uses=1]
	%tmp4650 = add i32 %tmp4644, %tmp46424643		; <i32> [#uses=1]
	%tmp4656 = add i32 %tmp4650, %tmp46484649		; <i32> [#uses=1]
	%tmp4657 = add i32 %tmp4656, %tmp46544655		; <i32> [#uses=2]
	%tmp4658 = ashr i32 %tmp4657, 2		; <i32> [#uses=1]
	%tmp4662 = load i16* null		; <i16> [#uses=1]
	%tmp46624663 = sext i16 %tmp4662 to i32		; <i32> [#uses=1]
	%tmp4672 = getelementptr %struct.BlockNode* null, i32 0, i32 1		; <i16*> [#uses=1]
	%tmp4673 = load i16* %tmp4672		; <i16> [#uses=1]
	%tmp46734674 = sext i16 %tmp4673 to i32		; <i32> [#uses=1]
	%tmp4678 = getelementptr %struct.BlockNode* null, i32 %tmp4254.sum, i32 1		; <i16*> [#uses=1]
	%tmp4679 = load i16* %tmp4678		; <i16> [#uses=1]
	%tmp46794680 = sext i16 %tmp4679 to i32		; <i32> [#uses=1]
	%tmp4669 = add i32 %tmp46624663, 2		; <i32> [#uses=1]
	%tmp4675 = add i32 %tmp4669, 0		; <i32> [#uses=1]
	%tmp4681 = add i32 %tmp4675, %tmp46734674		; <i32> [#uses=1]
	%tmp4682 = add i32 %tmp4681, %tmp46794680		; <i32> [#uses=2]
	%tmp4683 = ashr i32 %tmp4682, 2		; <i32> [#uses=1]
	%tmp4703 = load i32* %tmp21		; <i32> [#uses=1]
	%tmp4707 = shl i32 %tmp4703, 0		; <i32> [#uses=4]
	%tmp4710 = load %struct.BlockNode** null		; <%struct.BlockNode*> [#uses=6]
	%tmp4713 = mul i32 %tmp4707, %mb_y.4		; <i32> [#uses=1]
	%tmp4715 = add i32 %tmp4713, %mb_x.7		; <i32> [#uses=7]
	store i8 0, i8* null
	store i8 0, i8* null
	%tmp47594761 = bitcast %struct.BlockNode* null to i8*		; <i8*> [#uses=2]
	call void @llvm.memcpy.i32( i8* null, i8* %tmp47594761, i32 10, i32 0 )
	%tmp4716.sum5775 = add i32 %tmp4715, 1		; <i32> [#uses=1]
	%tmp4764 = getelementptr %struct.BlockNode* %tmp4710, i32 %tmp4716.sum5775		; <%struct.BlockNode*> [#uses=1]
	%tmp47644766 = bitcast %struct.BlockNode* %tmp4764 to i8*		; <i8*> [#uses=1]
	%tmp4716.sum5774 = add i32 %tmp4715, %tmp4707		; <i32> [#uses=0]
	%tmp47704772 = bitcast %struct.BlockNode* null to i8*		; <i8*> [#uses=1]
	%tmp4774 = add i32 %tmp4707, 1		; <i32> [#uses=1]
	%tmp4716.sum5773 = add i32 %tmp4774, %tmp4715		; <i32> [#uses=1]
	%tmp4777 = getelementptr %struct.BlockNode* %tmp4710, i32 %tmp4716.sum5773		; <%struct.BlockNode*> [#uses=1]
	%tmp47774779 = bitcast %struct.BlockNode* %tmp4777 to i8*		; <i8*> [#uses=1]
	%tmp4781 = icmp slt i32 %mb_x.7, 0		; <i1> [#uses=1]
	%tmp4788 = or i1 %tmp4781, %tmp4784		; <i1> [#uses=2]
	br i1 %tmp4788, label %cond_true4791, label %cond_next4794

cond_true4791:		; preds = %cond_next4622
	unreachable

cond_next4794:		; preds = %cond_next4622
	%tmp4797 = icmp slt i32 %mb_x.7, %tmp4707		; <i1> [#uses=1]
	br i1 %tmp4797, label %cond_next4803, label %cond_true4800

cond_true4800:		; preds = %cond_next4794
	unreachable

cond_next4803:		; preds = %cond_next4794
	%tmp4825 = ashr i32 %tmp4657, 12		; <i32> [#uses=1]
	shl i32 %tmp4682, 4		; <i32>:0 [#uses=1]
	%tmp4828 = and i32 %0, -64		; <i32> [#uses=1]
	%tmp4831 = getelementptr %struct.BlockNode* %tmp4710, i32 %tmp4715, i32 2		; <i8*> [#uses=0]
	%tmp4826 = add i32 %tmp4828, %tmp4825		; <i32> [#uses=1]
	%tmp4829 = add i32 %tmp4826, 0		; <i32> [#uses=1]
	%tmp4835 = add i32 %tmp4829, 0		; <i32> [#uses=1]
	store i32 %tmp4835, i32* null
	%tmp48534854 = trunc i32 %tmp4658 to i16		; <i16> [#uses=1]
	%tmp4856 = getelementptr %struct.BlockNode* %tmp4710, i32 %tmp4715, i32 0		; <i16*> [#uses=1]
	store i16 %tmp48534854, i16* %tmp4856
	%tmp48574858 = trunc i32 %tmp4683 to i16		; <i16> [#uses=1]
	%tmp4860 = getelementptr %struct.BlockNode* %tmp4710, i32 %tmp4715, i32 1		; <i16*> [#uses=1]
	store i16 %tmp48574858, i16* %tmp4860
	%tmp4866 = getelementptr %struct.BlockNode* %tmp4710, i32 %tmp4715, i32 4		; <i8*> [#uses=0]
	br i1 false, label %bb4933, label %cond_false4906

cond_false4906:		; preds = %cond_next4803
	call void @llvm.memcpy.i32( i8* %tmp47594761, i8* null, i32 10, i32 0 )
	call void @llvm.memcpy.i32( i8* %tmp47644766, i8* null, i32 10, i32 0 )
	call void @llvm.memcpy.i32( i8* %tmp47704772, i8* null, i32 10, i32 0 )
	call void @llvm.memcpy.i32( i8* %tmp47774779, i8* null, i32 10, i32 0 )
	br label %bb5215

bb4933:		; preds = %bb5215, %cond_next4803
	br i1 false, label %cond_true4944, label %bb5215

cond_true4944:		; preds = %bb4933
	%tmp4982 = load i32* %tmp21		; <i32> [#uses=1]
	%tmp4986 = shl i32 %tmp4982, 0		; <i32> [#uses=2]
	%tmp4992 = mul i32 %tmp4986, %mb_y.4		; <i32> [#uses=1]
	%tmp4994 = add i32 %tmp4992, %mb_x.7		; <i32> [#uses=5]
	%tmp4995.sum5765 = add i32 %tmp4994, 1		; <i32> [#uses=1]
	%tmp5043 = getelementptr %struct.BlockNode* null, i32 %tmp4995.sum5765		; <%struct.BlockNode*> [#uses=1]
	%tmp50435045 = bitcast %struct.BlockNode* %tmp5043 to i8*		; <i8*> [#uses=2]
	call void @llvm.memcpy.i32( i8* null, i8* %tmp50435045, i32 10, i32 0 )
	%tmp4995.sum5764 = add i32 %tmp4994, %tmp4986		; <i32> [#uses=1]
	%tmp5049 = getelementptr %struct.BlockNode* null, i32 %tmp4995.sum5764		; <%struct.BlockNode*> [#uses=1]
	%tmp50495051 = bitcast %struct.BlockNode* %tmp5049 to i8*		; <i8*> [#uses=2]
	call void @llvm.memcpy.i32( i8* null, i8* %tmp50495051, i32 10, i32 0 )
	%tmp4995.sum5763 = add i32 0, %tmp4994		; <i32> [#uses=1]
	%tmp5056 = getelementptr %struct.BlockNode* null, i32 %tmp4995.sum5763		; <%struct.BlockNode*> [#uses=1]
	%tmp50565058 = bitcast %struct.BlockNode* %tmp5056 to i8*		; <i8*> [#uses=1]
	br i1 %tmp4788, label %cond_true5070, label %cond_next5073

cond_true5070:		; preds = %cond_true4944
	unreachable

cond_next5073:		; preds = %cond_true4944
	%tmp5139 = getelementptr %struct.BlockNode* null, i32 %tmp4994, i32 1		; <i16*> [#uses=0]
	%tmp5145 = getelementptr %struct.BlockNode* null, i32 %tmp4994, i32 4		; <i8*> [#uses=0]
	call void @llvm.memcpy.i32( i8* %tmp50435045, i8* null, i32 10, i32 0 )
	call void @llvm.memcpy.i32( i8* %tmp50495051, i8* null, i32 10, i32 0 )
	call void @llvm.memcpy.i32( i8* %tmp50565058, i8* null, i32 10, i32 0 )
	br label %bb5215

bb5215:		; preds = %cond_next5073, %bb4933, %cond_false4906
	%i4232.3 = phi i32 [ 0, %cond_false4906 ], [ 0, %cond_next5073 ], [ 0, %bb4933 ]		; <i32> [#uses=1]
	%tmp5217 = icmp slt i32 %i4232.3, 4		; <i1> [#uses=1]
	br i1 %tmp5217, label %bb4933, label %bb5220

bb5220:		; preds = %bb5215
	br i1 false, label %bb5230, label %cond_true5226

cond_true5226:		; preds = %bb5220
	ret void

bb5230:		; preds = %bb5220
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb5233

bb5233:		; preds = %bb5233.preheader, %bb5230
	%indvar = phi i32 [ 0, %bb5233.preheader ], [ %indvar.next, %bb5230 ]		; <i32> [#uses=2]
	%mb_x.7 = shl i32 %indvar, 1		; <i32> [#uses=4]
	br i1 false, label %bb4231, label %bb5239

bb5239:		; preds = %bb5233
	%indvar.next37882 = add i32 %indvar37881, 1		; <i32> [#uses=1]
	br label %bb5242

bb5242:		; preds = %bb5239, %bb5242.preheader
	%indvar37881 = phi i32 [ 0, %bb5242.preheader ], [ %indvar.next37882, %bb5239 ]		; <i32> [#uses=2]
	%mb_y.4 = shl i32 %indvar37881, 1		; <i32> [#uses=3]
	br i1 false, label %bb5233.preheader, label %bb5248

bb5233.preheader:		; preds = %bb5242
	%tmp4784 = icmp slt i32 %mb_y.4, 0		; <i1> [#uses=1]
	br label %bb5233

bb5248:		; preds = %bb5242
	ret void

UnifiedReturnBlock:		; preds = %bb4221
	ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)
