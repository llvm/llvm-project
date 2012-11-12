; RUN: llc < %s	
%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.SYMBOL_TABLE_ENTRY = type { [9 x i8], [9 x i8], i32, i32, i32, %struct.SYMBOL_TABLE_ENTRY* }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
@str14 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]

declare void @fprintf(i32, ...)

define void @OUTPUT_TABLE(%struct.SYMBOL_TABLE_ENTRY* %SYM_TAB) {
entry:
	%tmp11 = getelementptr %struct.SYMBOL_TABLE_ENTRY* %SYM_TAB, i32 0, i32 1, i32 0		; <i8*> [#uses=2]
	%tmp.i = bitcast i8* %tmp11 to i8*		; <i8*> [#uses=1]
	br label %bb.i

bb.i:		; preds = %cond_next.i, %entry
	%s1.0.i = phi i8* [ %tmp.i, %entry ], [ null, %cond_next.i ]		; <i8*> [#uses=0]
	br i1 false, label %cond_true.i31, label %cond_next.i

cond_true.i31:		; preds = %bb.i
	call void (i32, ...)* @fprintf( i32 0, i8* %tmp11, i8* null )
	ret void

cond_next.i:		; preds = %bb.i
	br i1 false, label %bb.i, label %bb19.i

bb19.i:		; preds = %cond_next.i
	ret void
}
