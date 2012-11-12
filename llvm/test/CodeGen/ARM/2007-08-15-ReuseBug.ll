; RUN: llc < %s -mtriple=arm-apple-darwin -relocation-model=pic -mattr=+v6
; PR1609

	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
@_C_nextcmd = external global i32		; <i32*> [#uses=2]
@_C_cmds = external global [100 x i8*]		; <[100 x i8*]*> [#uses=2]
@.str44 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** %argv) {
entry:
	br label %cond_next212.i

bb21.i:		; preds = %cond_next212.i
	br label %cond_next212.i

bb24.i:		; preds = %cond_next212.i
	ret i32 0

bb27.i:		; preds = %cond_next212.i
	ret i32 0

bb30.i:		; preds = %cond_next212.i
	%tmp205399.i = add i32 %argc_addr.2358.0.i, -1		; <i32> [#uses=1]
	br label %cond_next212.i

bb33.i:		; preds = %cond_next212.i
	ret i32 0

cond_next73.i:		; preds = %cond_next212.i
	ret i32 0

bb75.i:		; preds = %cond_next212.i
	ret i32 0

bb77.i:		; preds = %cond_next212.i
	ret i32 0

bb79.i:		; preds = %cond_next212.i
	ret i32 0

bb102.i:		; preds = %cond_next212.i
	br i1 false, label %cond_true110.i, label %cond_next123.i

cond_true110.i:		; preds = %bb102.i
	%tmp116.i = getelementptr i8** %argv_addr.2321.0.i, i32 2		; <i8**> [#uses=1]
	%tmp117.i = load i8** %tmp116.i		; <i8*> [#uses=1]
	%tmp126425.i = call %struct.FILE* @fopen( i8* %tmp117.i, i8* getelementptr ([2 x i8]* @.str44, i32 0, i32 0) )		; <%struct.FILE*> [#uses=0]
	ret i32 0

cond_next123.i:		; preds = %bb102.i
	%tmp122.i = getelementptr i8* %tmp215.i, i32 2		; <i8*> [#uses=0]
	ret i32 0

bb162.i:		; preds = %cond_next212.i
	ret i32 0

C_addcmd.exit120.i:		; preds = %cond_next212.i
	%tmp3.i.i.i.i105.i = call i8* @calloc( i32 15, i32 1 )		; <i8*> [#uses=1]
	%tmp1.i108.i = getelementptr [100 x i8*]* @_C_cmds, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* %tmp3.i.i.i.i105.i, i8** %tmp1.i108.i, align 4
	%tmp.i91.i = load i32* @_C_nextcmd, align 4		; <i32> [#uses=1]
	store i32 0, i32* @_C_nextcmd, align 4
	%tmp3.i.i.i.i95.i = call i8* @calloc( i32 15, i32 1 )		; <i8*> [#uses=1]
	%tmp1.i98.i = getelementptr [100 x i8*]* @_C_cmds, i32 0, i32 %tmp.i91.i		; <i8**> [#uses=1]
	store i8* %tmp3.i.i.i.i95.i, i8** %tmp1.i98.i, align 4
	br label %cond_next212.i

bb174.i:		; preds = %cond_next212.i
	ret i32 0

bb192.i:		; preds = %cond_next212.i
	br label %cond_next212.i

cond_next212.i:		; preds = %cond_next212.i, %cond_next212.i, %cond_next212.i, %cond_next212.i, %bb192.i, %C_addcmd.exit120.i, %bb30.i, %bb21.i, %entry
	%max_d.3 = phi i32 [ -1, %entry ], [ %max_d.3, %bb30.i ], [ %max_d.3, %bb21.i ], [ %max_d.3, %C_addcmd.exit120.i ], [ 0, %bb192.i ], [ %max_d.3, %cond_next212.i ], [ %max_d.3, %cond_next212.i ], [ %max_d.3, %cond_next212.i ], [ %max_d.3, %cond_next212.i ]		; <i32> [#uses=7]
	%argv_addr.2321.0.i = phi i8** [ %argv, %entry ], [ %tmp214.i, %bb192.i ], [ %tmp214.i, %C_addcmd.exit120.i ], [ %tmp214.i, %bb30.i ], [ %tmp214.i, %bb21.i ], [ %tmp214.i, %cond_next212.i ], [ %tmp214.i, %cond_next212.i ], [ %tmp214.i, %cond_next212.i ], [ %tmp214.i, %cond_next212.i ]		; <i8**> [#uses=2]
	%argc_addr.2358.0.i = phi i32 [ %argc, %entry ], [ %tmp205399.i, %bb30.i ], [ 0, %bb21.i ], [ 0, %C_addcmd.exit120.i ], [ 0, %bb192.i ], [ 0, %cond_next212.i ], [ 0, %cond_next212.i ], [ 0, %cond_next212.i ], [ 0, %cond_next212.i ]		; <i32> [#uses=1]
	%tmp214.i = getelementptr i8** %argv_addr.2321.0.i, i32 1		; <i8**> [#uses=9]
	%tmp215.i = load i8** %tmp214.i		; <i8*> [#uses=1]
	%tmp1314.i = sext i8 0 to i32		; <i32> [#uses=1]
	switch i32 %tmp1314.i, label %bb192.i [
		 i32 76, label %C_addcmd.exit120.i
		 i32 77, label %bb174.i
		 i32 83, label %bb162.i
		 i32 97, label %bb33.i
		 i32 98, label %bb21.i
		 i32 99, label %bb24.i
		 i32 100, label %bb27.i
		 i32 101, label %cond_next212.i
		 i32 102, label %bb102.i
		 i32 105, label %bb75.i
		 i32 109, label %bb30.i
		 i32 113, label %cond_next212.i
		 i32 114, label %cond_next73.i
		 i32 115, label %bb79.i
		 i32 116, label %cond_next212.i
		 i32 118, label %bb77.i
		 i32 119, label %cond_next212.i
	]
}

declare %struct.FILE* @fopen(i8*, i8*)

declare i8* @calloc(i32, i32)
