; RUN: llvm-as < %s | llc
; PR2096
	%struct.AVClass = type { i8*, i8* (i8*)*, %struct.AVOption* }
	%struct.AVCodec = type { i8*, i32, i32, i32, i32 (%struct.AVCodecContext*)*, i32 (%struct.AVCodecContext*, i8*, i32, i8*)*, i32 (%struct.AVCodecContext*)*, i32 (%struct.AVCodecContext*, i8*, i32*, i8*, i32)*, i32, %struct.AVCodec*, void (%struct.AVCodecContext*)*, %struct.AVRational*, i32* }
	%struct.AVCodecContext = type { %struct.AVClass*, i32, i32, i32, i32, i32, i8*, i32, %struct.AVRational, i32, i32, i32, i32, i32, void (%struct.AVCodecContext*, %struct.AVFrame*, i32*, i32, i32, i32)*, i32, i32, i32, i32, i32, i32, i32, float, float, i32, i32, i32, i32, float, i32, i32, i32, %struct.AVCodec*, i8*, i32, i32, void (%struct.AVCodecContext*, i8*, i32, i32)*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, [32 x i8], i32, i32, i32, i32, i32, i32, i32, float, i32, i32 (%struct.AVCodecContext*, %struct.AVFrame*)*, void (%struct.AVCodecContext*, %struct.AVFrame*)*, i32, i32, i32, i32, i8*, i8*, float, float, i32, %struct.RcOverride*, i32, i8*, i32, i32, i32, float, float, float, float, i32, float, float, float, float, float, i32, i32, i32, i32*, i32, i32, i32, i32, %struct.AVRational, %struct.AVFrame*, i32, i32, [4 x i64], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 (%struct.AVCodecContext*, i32*)*, i32, i32, i32, i32, i32, i32, i8*, i32, i32, i32, i32, i32, i32, i16*, i16*, i32, i32, i32, i32, %struct.AVPaletteControl*, i32, i32 (%struct.AVCodecContext*, %struct.AVFrame*)*, i32, i32, i32, i32, i32, i32, i32, i32 (%struct.AVCodecContext*, i32 (%struct.AVCodecContext*, i8*)*, i8**, i32*, i32)*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, float }
	%struct.AVFrame = type { [4 x i8*], [4 x i32], [4 x i8*], i32, i32, i64, i32, i32, i32, i32, i32, i8*, i32, i8*, [2 x [2 x i16]*], i32*, i8, i8*, [4 x i64], i32, i32, i32, i32, i32, %struct.AVPanScan*, i32, i32, i16*, [2 x i8*] }
	%struct.AVOption = type opaque
	%struct.AVPaletteControl = type { i32, [256 x i32] }
	%struct.AVPanScan = type { i32, i32, i32, [3 x [2 x i16]] }
	%struct.AVRational = type { i32, i32 }
	%struct.RcOverride = type { i32, i32, i32, float }

define i32 @sonic_encode_frame(%struct.AVCodecContext* %avctx, i8* %buf, i32 %buf_size, i8* %data) {
entry:
	switch i32 0, label %bb429 [
		 i32 0, label %bb244.preheader
		 i32 1, label %bb279.preheader
	]

bb279.preheader:		; preds = %entry
	ret i32 0

bb244.preheader:		; preds = %entry
	ret i32 0

bb429:		; preds = %entry
	br i1 false, label %bb.nph1770, label %bb627

bb.nph1770:		; preds = %bb429
	br i1 false, label %bb471, label %bb505

bb471:		; preds = %bb471, %bb.nph1770
	%tmp487 = add double 0.000000e+00, 0.000000e+00		; <double> [#uses=1]
	br i1 false, label %bb505, label %bb471

bb505:		; preds = %bb471, %bb.nph1770
	%xy.0.lcssa = phi double [ 0.000000e+00, %bb.nph1770 ], [ %tmp487, %bb471 ]		; <double> [#uses=1]
	%tmp507 = sub double -0.000000e+00, %xy.0.lcssa		; <double> [#uses=1]
	%tmp509 = fdiv double %tmp507, 0.000000e+00		; <double> [#uses=1]
	%tmp510 = mul double %tmp509, 1.024000e+03		; <double> [#uses=1]
	%tmp516 = fdiv double %tmp510, 0.000000e+00		; <double> [#uses=1]
	%tmp517 = add double %tmp516, 5.000000e-01		; <double> [#uses=1]
	%tmp518 = tail call double @floor( double %tmp517 ) nounwind readnone 		; <double> [#uses=0]
	ret i32 0

bb627:		; preds = %bb429
	ret i32 0
}

declare double @floor(double) nounwind readnone 
