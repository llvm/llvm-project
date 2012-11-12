; PR 1200
; RUN: llc < %s -enable-tail-merge=0 | not grep jmp 

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8"
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.Index_Map = type { i32, %struct.item_set** }
	%struct.Item = type { [4 x i16], %struct.rule* }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.dimension = type { i16*, %struct.Index_Map, %struct.mapping*, i32, %struct.plankMap* }
	%struct.item_set = type { i32, i32, %struct.operator*, [2 x %struct.item_set*], %struct.item_set*, i16*, %struct.Item*, %struct.Item* }
	%struct.list = type { i8*, %struct.list* }
	%struct.mapping = type { %struct.list**, i32, i32, i32, %struct.item_set** }
	%struct.nonterminal = type { i8*, i32, i32, i32, %struct.plankMap*, %struct.rule* }
	%struct.operator = type { i8*, i8, i32, i32, i32, i32, %struct.table* }
	%struct.pattern = type { %struct.nonterminal*, %struct.operator*, [2 x %struct.nonterminal*] }
	%struct.plank = type { i8*, %struct.list*, i32 }
	%struct.plankMap = type { %struct.list*, i32, %struct.stateMap* }
	%struct.rule = type { [4 x i16], i32, i32, i32, %struct.nonterminal*, %struct.pattern*, i8 }
	%struct.stateMap = type { i8*, %struct.plank*, i32, i16* }
	%struct.table = type { %struct.operator*, %struct.list*, i16*, [2 x %struct.dimension*], %struct.item_set** }
@outfile = external global %struct.FILE*		; <%struct.FILE**> [#uses=1]
@str1 = external global [11 x i8]		; <[11 x i8]*> [#uses=1]

declare i32 @fprintf(%struct.FILE*, i8*, ...)

define i16 @main_bb_2E_i9_2E_i_2E_i932_2E_ce(%struct.list* %l_addr.01.0.i2.i.i929, %struct.operator** %tmp66.i62.i.out) {
newFuncRoot:
	br label %bb.i9.i.i932.ce

NewDefault:		; preds = %LeafBlock, %LeafBlock1, %LeafBlock2, %LeafBlock3
	br label %bb36.i.i.exitStub

bb36.i.i.exitStub:		; preds = %NewDefault
	store %struct.operator* %tmp66.i62.i, %struct.operator** %tmp66.i62.i.out
	ret i16 0

bb.i14.i.exitStub:		; preds = %LeafBlock
	store %struct.operator* %tmp66.i62.i, %struct.operator** %tmp66.i62.i.out
	ret i16 1

bb12.i.i935.exitStub:		; preds = %LeafBlock1
	store %struct.operator* %tmp66.i62.i, %struct.operator** %tmp66.i62.i.out
	ret i16 2

bb20.i.i937.exitStub:		; preds = %LeafBlock2
	store %struct.operator* %tmp66.i62.i, %struct.operator** %tmp66.i62.i.out
	ret i16 3

bb28.i.i938.exitStub:		; preds = %LeafBlock3
	store %struct.operator* %tmp66.i62.i, %struct.operator** %tmp66.i62.i.out
	ret i16 4

bb.i9.i.i932.ce:		; preds = %newFuncRoot
	%tmp1.i3.i.i930 = getelementptr %struct.list* %l_addr.01.0.i2.i.i929, i32 0, i32 0		; <i8**> [#uses=1]
	%tmp2.i4.i.i931 = load i8** %tmp1.i3.i.i930		; <i8*> [#uses=1]
	%tmp66.i62.i = bitcast i8* %tmp2.i4.i.i931 to %struct.operator*		; <%struct.operator*> [#uses=7]
	%tmp1.i6.i = getelementptr %struct.operator* %tmp66.i62.i, i32 0, i32 2		; <i32*> [#uses=1]
	%tmp2.i7.i = load i32* %tmp1.i6.i		; <i32> [#uses=1]
	%tmp3.i8.i = load %struct.FILE** @outfile		; <%struct.FILE*> [#uses=1]
	%tmp5.i9.i = call i32 (%struct.FILE*, i8*, ...)* @fprintf( %struct.FILE* %tmp3.i8.i, i8* getelementptr ([11 x i8]* @str1, i32 0, i32 0), i32 %tmp2.i7.i )		; <i32> [#uses=0]
	%tmp7.i10.i = getelementptr %struct.operator* %tmp66.i62.i, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp8.i11.i = load i32* %tmp7.i10.i		; <i32> [#uses=7]
	br label %NodeBlock5

NodeBlock5:		; preds = %bb.i9.i.i932.ce
	icmp slt i32 %tmp8.i11.i, 1		; <i1>:0 [#uses=1]
	br i1 %0, label %NodeBlock, label %NodeBlock4

NodeBlock4:		; preds = %NodeBlock5
	icmp slt i32 %tmp8.i11.i, 2		; <i1>:1 [#uses=1]
	br i1 %1, label %LeafBlock2, label %LeafBlock3

LeafBlock3:		; preds = %NodeBlock4
	icmp eq i32 %tmp8.i11.i, 2		; <i1>:2 [#uses=1]
	br i1 %2, label %bb28.i.i938.exitStub, label %NewDefault

LeafBlock2:		; preds = %NodeBlock4
	icmp eq i32 %tmp8.i11.i, 1		; <i1>:3 [#uses=1]
	br i1 %3, label %bb20.i.i937.exitStub, label %NewDefault

NodeBlock:		; preds = %NodeBlock5
	icmp slt i32 %tmp8.i11.i, 0		; <i1>:4 [#uses=1]
	br i1 %4, label %LeafBlock, label %LeafBlock1

LeafBlock1:		; preds = %NodeBlock
	icmp eq i32 %tmp8.i11.i, 0		; <i1>:5 [#uses=1]
	br i1 %5, label %bb12.i.i935.exitStub, label %NewDefault

LeafBlock:		; preds = %NodeBlock
	icmp eq i32 %tmp8.i11.i, -1		; <i1>:6 [#uses=1]
	br i1 %6, label %bb.i14.i.exitStub, label %NewDefault
}
