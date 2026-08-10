; RUN: llc < %s	
%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.SYMBOL_TABLE_ENTRY = type { [9 x i8], [9 x i8], i32, i32, i32, ptr }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
@str14 = external global [6 x i8]		; <ptr> [#uses=0]

declare void @fprintf(i32, ...)

define void @OUTPUT_TABLE(ptr %SYM_TAB) {
entry:
	%tmp11 = getelementptr %struct.SYMBOL_TABLE_ENTRY, ptr %SYM_TAB, i32 0, i32 1, i32 0		; <ptr> [#uses=2]
	br label %bb.i

bb.i:		; preds = %cond_next.i, %entry
	%s1.0.i = phi ptr [ %tmp11, %entry ], [ null, %cond_next.i ]		; <ptr> [#uses=0]
	br i1 false, label %cond_true.i31, label %cond_next.i

cond_true.i31:		; preds = %bb.i
	call void (i32, ...) @fprintf( i32 0, ptr %tmp11, ptr null )
	ret void

cond_next.i:		; preds = %bb.i
	br i1 false, label %bb.i, label %bb19.i

bb19.i:		; preds = %cond_next.i
	ret void
}
