; RUN: llvm-as < %s | llc

	%struct.AVClass = type { i8*, i8* (i8*)*, %struct.AVOption* }
	%struct.AVCodec = type { i8*, i32, i32, i32, i32 (%struct.AVCodecContext*)*, i32 (%struct.AVCodecContext*, i8*, i32, i8*)*, i32 (%struct.AVCodecContext*)*, i32 (%struct.AVCodecContext*, i8*, i32*, i8*, i32)*, i32, %struct.AVCodec*, void (%struct.AVCodecContext*)*, %struct.AVRational*, i32* }
	%struct.AVCodecContext = type { %struct.AVClass*, i32, i32, i32, i32, i32, i8*, i32, %struct.AVRational, i32, i32, i32, i32, i32, void (%struct.AVCodecContext*, %struct.AVFrame*, i32*, i32, i32, i32)*, i32, i32, i32, i32, i32, i32, i32, float, float, i32, i32, i32, i32, float, i32, i32, i32, %struct.AVCodec*, i8*, i32, i32, void (%struct.AVCodecContext*, i8*, i32, i32)*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, [32 x i8], i32, i32, i32, i32, i32, i32, i32, float, i32, i32 (%struct.AVCodecContext*, %struct.AVFrame*)*, void (%struct.AVCodecContext*, %struct.AVFrame*)*, i32, i32, i32, i32, i8*, i8*, float, float, i32, %struct.RcOverride*, i32, i8*, i32, i32, i32, float, float, float, float, i32, float, float, float, float, float, i32, i32, i32, i32*, i32, i32, i32, i32, %struct.AVRational, %struct.AVFrame*, i32, i32, [4 x i64], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 (%struct.AVCodecContext*, i32*)*, i32, i32, i32, i32, i32, i32, i8*, i32, i32, i32, i32, i32, i32, i16*, i16*, i32, i32, i32, i32, %struct.AVPaletteControl*, i32, i32 (%struct.AVCodecContext*, %struct.AVFrame*)*, i32, i32, i32, i32, i32, i32, i32, i32 (%struct.AVCodecContext*, i32 (%struct.AVCodecContext*, i8*)*, i8**, i32*, i32)*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64 }
	%struct.AVEvalExpr = type opaque
	%struct.AVFrame = type { [4 x i8*], [4 x i32], [4 x i8*], i32, i32, i64, i32, i32, i32, i32, i32, i8*, i32, i8*, [2 x [2 x i16]*], i32*, i8, i8*, [4 x i64], i32, i32, i32, i32, i32, %struct.AVPanScan*, i32, i32, i16*, [2 x i8*] }
	%struct.AVOption = type opaque
	%struct.AVPaletteControl = type { i32, [256 x i32] }
	%struct.AVPanScan = type { i32, i32, i32, [3 x [2 x i16]] }
	%struct.AVRational = type { i32, i32 }
	%struct.DSPContext = type { void (i16*, i8*, i32)*, void (i16*, i8*, i8*, i32)*, void (i16*, i8*, i32)*, void (i16*, i8*, i32)*, void (i16*, i8*, i32)*, void (i8*, i16*, i32)*, void (i8*, i16*, i32)*, void (i8*, i8*, i32, i32, i32, i32, i32)*, void (i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)*, void (i16*)*, i32 (i8*, i32)*, i32 (i8*, i32)*, [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], [5 x i32 (i8*, i8*, i8*, i32, i32)*], i32 (i8*, i16*, i32)*, [4 x [4 x void (i8*, i8*, i32, i32)*]], [4 x [4 x void (i8*, i8*, i32, i32)*]], [4 x [4 x void (i8*, i8*, i32, i32)*]], [4 x [4 x void (i8*, i8*, i32, i32)*]], [2 x void (i8*, i8*, i8*, i32, i32)*], [11 x void (i8*, i8*, i32, i32, i32)*], [11 x void (i8*, i8*, i32, i32, i32)*], [2 x [16 x void (i8*, i8*, i32)*]], [2 x [16 x void (i8*, i8*, i32)*]], [2 x [16 x void (i8*, i8*, i32)*]], [2 x [16 x void (i8*, i8*, i32)*]], [8 x void (i8*, i8*, i32)*], [3 x void (i8*, i8*, i32, i32, i32, i32)*], [3 x void (i8*, i8*, i32, i32, i32, i32)*], [3 x void (i8*, i8*, i32, i32, i32, i32)*], [4 x [16 x void (i8*, i8*, i32)*]], [4 x [16 x void (i8*, i8*, i32)*]], [4 x [16 x void (i8*, i8*, i32)*]], [4 x [16 x void (i8*, i8*, i32)*]], [10 x void (i8*, i32, i32, i32, i32)*], [10 x void (i8*, i8*, i32, i32, i32, i32, i32)*], [2 x [16 x void (i8*, i8*, i32)*]], [2 x [16 x void (i8*, i8*, i32)*]], void (i8*, i32, i32, i32, i32, i32, i32)*, void (i8*, i32, i32, i32, i32, i32, i32)*, void (i8*, i32, i32, i32, i32, i32, i32)*, void (i8*, i32, i32, i32, i32, i32, i32)*, void (i8*, i16*, i32)*, [2 x [4 x i32 (i8*, i8*, i8*, i32, i32)*]], void (i8*, i8*, i32)*, void (i8*, i8*, i8*, i32)*, void (i8*, i8*, i8*, i32, i32*, i32*)*, void (i32*, i32*, i32)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32)*, void (i8*, i32, i32, i32)*, void ([4 x [4 x i16]]*, i8*, [40 x i8]*, [40 x [2 x i16]]*, i32, i32, i32, i32, i32)*, void (i8*, i32, i32)*, void (i8*, i32, i32)*, void (i8*, i32)*, void (float*, float*, i32)*, void (float*, float*, i32)*, void (float*, float*, float*, i32)*, void (float*, float*, float*, float*, i32, i32, i32)*, void (i16*, float*, i32)*, void (i16*)*, void (i16*)*, void (i16*)*, void (i8*, i32, i16*)*, void (i8*, i32, i16*)*, [64 x i8], i32, i32 (i16*, i16*, i16*, i32)*, void (i16*, i16*, i32)*, void (i8*, i16*, i32)*, void (i8*, i16*, i32)*, void (i8*, i16*, i32)*, void (i8*, i16*, i32)*, void ([4 x i16]*)*, void (i32*, i32*, i32*, i32*, i32*, i32*, i32)*, void (i32*, i32)*, void (i8*, i32, i8**, i32, i32, i32, i32, i32, %struct.slice_buffer*, i32, i8*)*, void (i8*, i32, i32)*, [4 x void (i8*, i32, i8*, i32, i32, i32)*], void (i16*)*, void (i16*, i32)*, void (i16*, i32)*, void (i16*, i32)*, void (i8*, i32)*, void (i8*, i32)*, [16 x void (i8*, i8*, i32, i32)*] }
	%struct.FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct.GetBitContext = type { i8*, i8*, i32*, i32, i32, i32, i32 }
	%struct.MJpegContext = type opaque
	%struct.MotionEstContext = type { %struct.AVCodecContext*, i32, [4 x [2 x i32]], [4 x [2 x i32]], i8*, i8*, [2 x i8*], i8*, i32, i32*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [4 x [4 x i8*]], [4 x [4 x i8*]], i32, i32, i32, i32, i32, [4 x void (i8*, i8*, i32, i32)*]*, [4 x void (i8*, i8*, i32, i32)*]*, [16 x void (i8*, i8*, i32)*]*, [16 x void (i8*, i8*, i32)*]*, [4097 x i8]*, i8*, i32 (%struct.MpegEncContext*, i32*, i32*, i32, i32, i32, i32, i32)* }
	%struct.MpegEncContext = type { %struct.AVCodecContext*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.PutBitContext, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.Picture*, %struct.Picture**, %struct.Picture**, i32, i32, [8 x %struct.MpegEncContext*], %struct.Picture, %struct.Picture, %struct.Picture, %struct.Picture, %struct.Picture*, %struct.Picture*, %struct.Picture*, [3 x i8*], [3 x i32], i16*, [3 x i16*], [20 x i16], i32, i32, i8*, i8*, i8*, i8*, i8*, [16 x i16]*, [3 x [16 x i16]*], i32, i8*, i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, [5 x i32], i32, i32, i32, i32, %struct.DSPContext, i32, i32, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x [2 x [2 x i16]*]], [2 x [2 x [2 x [2 x i16]*]]], [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x i16]*, [2 x [2 x [2 x i16]*]], [2 x [2 x [2 x [2 x i16]*]]], [2 x i8*], [2 x [2 x i8*]], i32, i32, i32, [2 x [4 x [2 x i32]]], [2 x [2 x i32]], [2 x [2 x [2 x i32]]], i8*, [2 x [64 x i16]], %struct.MotionEstContext, i32, i32, i32, i32, i32, i32, i16*, [6 x i32], [6 x i32], [3 x i8*], i32*, [64 x i16], [64 x i16], [64 x i16], [64 x i16], i32, i32, i32, i32, i32, i8*, i8*, i8*, i8*, i8*, i8*, [8 x i32], [64 x i32]*, [64 x i32]*, [2 x [64 x i16]]*, [2 x [64 x i16]]*, [12 x i32], %struct.ScanTable, %struct.ScanTable, %struct.ScanTable, %struct.ScanTable, [64 x i32]*, [2 x i32], [64 x i16]*, i8*, i64, i64, i32, i32, %struct.RateControlContext, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i32, i32, %struct.GetBitContext, i32, i32, i32, %struct.ParseContext, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i16, i16, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x [2 x i32]], [2 x [2 x i32]], [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.PutBitContext, %struct.PutBitContext, i32, i32, i32, i32, i32, i32, i8*, i32, i32, i32, i32, i32, [3 x i32], %struct.MJpegContext*, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x [65 x [65 x [2 x i32]]]]*, i32, i32, %struct.GetBitContext, i32, i32, i32, i8*, i32, [2 x [2 x i32]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, i32, i32, i32, i8*, i32, [12 x i16*], [64 x i16]*, [8 x [64 x i16]]*, i32 (%struct.MpegEncContext*, [64 x i16]*)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, void (%struct.MpegEncContext*, i16*, i32, i32)*, i32 (%struct.MpegEncContext*, i16*, i32, i32, i32*)*, i32 (%struct.MpegEncContext*, i16*, i32, i32, i32*)*, void (%struct.MpegEncContext*, i16*)* }
	%struct.ParseContext = type { i8*, i32, i32, i32, i32, i32, i32, i32 }
	%struct.Picture = type { [4 x i8*], [4 x i32], [4 x i8*], i32, i32, i64, i32, i32, i32, i32, i32, i8*, i32, i8*, [2 x [2 x i16]*], i32*, i8, i8*, [4 x i64], i32, i32, i32, i32, i32, %struct.AVPanScan*, i32, i32, i16*, [2 x i8*], [3 x i8*], [2 x [2 x i16]*], i32*, [2 x i32], i32, i32, i32, i32, [2 x [16 x i32]], [2 x i32], i32, i32, i16*, i16*, i8*, i32*, i32 }
	%struct.Predictor = type { double, double, double }
	%struct.PutBitContext = type { i32, i32, i8*, i8*, i8* }
	%struct.RateControlContext = type { %struct.FILE*, i32, %struct.RateControlEntry*, double, [5 x %struct.Predictor], double, double, double, double, double, [5 x double], i32, i32, [5 x i64], [5 x i64], [5 x i64], [5 x i64], [5 x i32], i32, i8*, float, i32, %struct.AVEvalExpr* }
	%struct.RateControlEntry = type { i32, float, i32, i32, i32, i32, i32, i64, i32, float, i32, i32, i32, i32, i32, i32 }
	%struct.RcOverride = type { i32, i32, i32, float }
	%struct.ScanTable = type { i8*, [64 x i8], [64 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, i32 }
	%struct.slice_buffer = type opaque

define float @ff_rate_estimate_qscale(%struct.MpegEncContext* %s, i32 %dry_run) {
entry:
	br i1 false, label %cond_false163, label %cond_true135

cond_true135:		; preds = %entry
	ret float 0.000000e+00

cond_false163:		; preds = %entry
	br i1 false, label %cond_true203, label %cond_next211

cond_true203:		; preds = %cond_false163
	ret float 0.000000e+00

cond_next211:		; preds = %cond_false163
	br i1 false, label %cond_false243, label %cond_true220

cond_true220:		; preds = %cond_next211
	br i1 false, label %cond_next237, label %cond_true225

cond_true225:		; preds = %cond_true220
	ret float 0.000000e+00

cond_next237:		; preds = %cond_true220
	br i1 false, label %cond_false785, label %cond_true735

cond_false243:		; preds = %cond_next211
	ret float 0.000000e+00

cond_true735:		; preds = %cond_next237
	ret float 0.000000e+00

cond_false785:		; preds = %cond_next237
	br i1 false, label %cond_true356.i.preheader, label %bb359.i

cond_true356.i.preheader:		; preds = %cond_false785
	%tmp116117.i = zext i8 0 to i32		; <i32> [#uses=1]
	br i1 false, label %cond_false.i, label %cond_next159.i

cond_false.i:		; preds = %cond_true356.i.preheader
	ret float 0.000000e+00

cond_next159.i:		; preds = %cond_true356.i.preheader
	%tmp178.i = add i32 %tmp116117.i, -128		; <i32> [#uses=2]
	%tmp181.i = mul i32 %tmp178.i, %tmp178.i		; <i32> [#uses=1]
	%tmp181182.i = sitofp i32 %tmp181.i to float		; <float> [#uses=1]
	%tmp199200.pn.in.i = mul float %tmp181182.i, 0.000000e+00		; <float> [#uses=1]
	%tmp199200.pn.i = fpext float %tmp199200.pn.in.i to double		; <double> [#uses=1]
	%tmp201.pn.i = sub double 1.000000e+00, %tmp199200.pn.i		; <double> [#uses=1]
	%factor.2.in.i = mul double 0.000000e+00, %tmp201.pn.i		; <double> [#uses=1]
	%factor.2.i = fptrunc double %factor.2.in.i to float		; <float> [#uses=1]
	br i1 false, label %cond_next312.i, label %cond_false222.i

cond_false222.i:		; preds = %cond_next159.i
	ret float 0.000000e+00

cond_next312.i:		; preds = %cond_next159.i
	%tmp313314.i = fpext float %factor.2.i to double		; <double> [#uses=0]
	ret float 0.000000e+00

bb359.i:		; preds = %cond_false785
	ret float 0.000000e+00
}
