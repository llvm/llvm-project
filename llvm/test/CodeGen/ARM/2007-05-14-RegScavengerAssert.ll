; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR1406

	%struct.AVClass = type { i8*, i8* (i8*)*, %struct.AVOption* }
	%struct.AVCodec = type { i8*, i32, i32, i32, i32 (%struct.AVCodecContext*)*, i32 (%struct.AVCodecContext*, i8*, i32, i8*)*, i32 (%struct.AVCodecContext*)*, i32 (%struct.AVCodecContext*, i8*, i32*, i8*, i32)*, i32, %struct.AVCodec*, void (%struct.AVCodecContext*)*, %struct.AVRational*, i32* }
	%struct.AVCodecContext = type { %struct.AVClass*, i32, i32, i32, i32, i32, i8*, i32, %struct.AVRational, i32, i32, i32, i32, i32, void (%struct.AVCodecContext*, %struct.AVFrame*, i32*, i32, i32, i32)*, i32, i32, i32, i32, i32, i32, i32, float, float, i32, i32, i32, i32, float, i32, i32, i32, %struct.AVCodec*, i8*, i32, i32, void (%struct.AVCodecContext*, i8*, i32, i32)*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, [32 x i8], i32, i32, i32, i32, i32, i32, i32, float, i32, i32 (%struct.AVCodecContext*, %struct.AVFrame*)*, void (%struct.AVCodecContext*, %struct.AVFrame*)*, i32, i32, i32, i32, i8*, i8*, float, float, i32, %struct.RcOverride*, i32, i8*, i32, i32, i32, float, float, float, float, i32, float, float, float, float, float, i32, i32, i32, i32*, i32, i32, i32, i32, %struct.AVRational, %struct.AVFrame*, i32, i32, [4 x i64], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 (%struct.AVCodecContext*, i32*)*, i32, i32, i32, i32, i32, i32, i8*, i32, i32, i32, i32, i32, i32, i16*, i16*, i32, i32, i32, i32, %struct.AVPaletteControl*, i32, i32 (%struct.AVCodecContext*, %struct.AVFrame*)*, i32, i32, i32, i32, i32, i32, i32, i32 (%struct.AVCodecContext*, i32 (%struct.AVCodecContext*, i8*)*, i8**, i32*, i32)*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64 }
	%struct.AVFrame = type { [4 x i8*], [4 x i32], [4 x i8*], i32, i32, i64, i32, i32, i32, i32, i32, i8*, i32, i8*, [2 x [2 x i16]*], i32*, i8, i8*, [4 x i64], i32, i32, i32, i32, i32, %struct.AVPanScan*, i32, i32, i16*, [2 x i8*] }
	%struct.AVOption = type opaque
	%struct.AVPaletteControl = type { i32, [256 x i32] }
	%struct.AVPanScan = type { i32, i32, i32, [3 x [2 x i16]] }
	%struct.AVRational = type { i32, i32 }
	%struct.RcOverride = type { i32, i32, i32, float }

define i32 @decode_init(%struct.AVCodecContext* %avctx) {
entry:
	br i1 false, label %bb, label %cond_next789

bb:		; preds = %bb, %entry
	br i1 false, label %bb59, label %bb

bb59:		; preds = %bb
	%tmp68 = sdiv i64 0, 0		; <i64> [#uses=1]
	%tmp6869 = trunc i64 %tmp68 to i32		; <i32> [#uses=2]
	%tmp81 = call i32 asm "smull $0, $1, $2, $3     \0A\09mov   $0, $0,     lsr $4\0A\09add   $1, $0, $1, lsl $5\0A\09", "=&r,=*&r,r,r,i,i"( i32* null, i32 %tmp6869, i32 13316085, i32 23, i32 9 )		; <i32> [#uses=0]
	%tmp90 = call i32 asm "smull $0, $1, $2, $3     \0A\09mov   $0, $0,     lsr $4\0A\09add   $1, $0, $1, lsl $5\0A\09", "=&r,=*&r,r,r,i,i"( i32* null, i32 %tmp6869, i32 10568984, i32 23, i32 9 )		; <i32> [#uses=0]
	unreachable

cond_next789:		; preds = %entry
	ret i32 0
}
