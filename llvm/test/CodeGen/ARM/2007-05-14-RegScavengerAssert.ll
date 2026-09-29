; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR1406

	%struct.AVClass = type { ptr, ptr, ptr }
	%struct.AVCodec = type { ptr, i32, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr }
	%struct.AVCodecContext = type { ptr, i32, i32, i32, i32, i32, ptr, i32, %struct.AVRational, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, float, float, i32, i32, i32, i32, float, i32, i32, i32, ptr, ptr, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, [32 x i8], i32, i32, i32, i32, i32, i32, i32, float, i32, ptr, ptr, i32, i32, i32, i32, ptr, ptr, float, float, i32, ptr, i32, ptr, i32, i32, i32, float, float, float, float, i32, float, float, float, float, float, i32, i32, i32, ptr, i32, i32, i32, i32, %struct.AVRational, ptr, i32, i32, [4 x i64], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, i32, i32, i32, ptr, i32, ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64 }
	%struct.AVFrame = type { [4 x ptr], [4 x i32], [4 x ptr], i32, i32, i64, i32, i32, i32, i32, i32, ptr, i32, ptr, [2 x ptr], ptr, i8, ptr, [4 x i64], i32, i32, i32, i32, i32, ptr, i32, i32, ptr, [2 x ptr] }
	%struct.AVOption = type opaque
	%struct.AVPaletteControl = type { i32, [256 x i32] }
	%struct.AVPanScan = type { i32, i32, i32, [3 x [2 x i16]] }
	%struct.AVRational = type { i32, i32 }
	%struct.RcOverride = type { i32, i32, i32, float }

define i32 @decode_init(ptr %avctx) {
entry:
	br i1 false, label %bb, label %cond_next789

bb:		; preds = %bb, %entry
	br i1 false, label %bb59, label %bb

bb59:		; preds = %bb
	%tmp68 = sdiv i64 0, 0		; <i64> [#uses=1]
	%tmp6869 = trunc i64 %tmp68 to i32		; <i32> [#uses=2]
	%tmp81 = call i32 asm "smull $0, $1, $2, $3     \0A\09mov   $0, $0,     lsr $4\0A\09add   $1, $0, $1, lsl $5\0A\09", "=&r,=*&r,r,r,i,i"( ptr elementtype( i32) null, i32 %tmp6869, i32 13316085, i32 23, i32 9 )		; <i32> [#uses=0]
	%tmp90 = call i32 asm "smull $0, $1, $2, $3     \0A\09mov   $0, $0,     lsr $4\0A\09add   $1, $0, $1, lsl $5\0A\09", "=&r,=*&r,r,r,i,i"( ptr elementtype( i32) null, i32 %tmp6869, i32 10568984, i32 23, i32 9 )		; <i32> [#uses=0]
	unreachable

cond_next789:		; preds = %entry
	ret i32 0
}
