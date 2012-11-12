; RUN: llc < %s  -mtriple=x86_64-linux -mcpu=corei7 | grep pinsr
; PR2562

external global i16		; <i16*>:0 [#uses=1]
external global <4 x i16>		; <<4 x i16>*>:1 [#uses=2]

declare void @abort()

define void @""() {
	load i16* @0		; <i16>:1 [#uses=1]
	load <4 x i16>* @1		; <<4 x i16>>:2 [#uses=1]
	insertelement <4 x i16> %2, i16 %1, i32 0		; <<4 x i16>>:3 [#uses=1]
	store <4 x i16> %3, <4 x i16>* @1
	ret void
}
