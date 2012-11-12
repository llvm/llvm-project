; PR925
; RUN: llc < %s -march=x86 | \
; RUN:   grep mov.*str1 | count 1

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8.7.2"
@str1 = internal constant [5 x i8] c"bonk\00"		; <[5 x i8]*> [#uses=1]
@str2 = internal constant [5 x i8] c"bork\00"		; <[5 x i8]*> [#uses=1]
@str = internal constant [8 x i8] c"perfwap\00"		; <[8 x i8]*> [#uses=1]

define void @foo(i32 %C) {
entry:
	switch i32 %C, label %bb2 [
		 i32 1, label %blahaha
		 i32 2, label %blahaha
		 i32 3, label %blahaha
		 i32 4, label %blahaha
		 i32 5, label %blahaha
		 i32 6, label %blahaha
		 i32 7, label %blahaha
		 i32 8, label %blahaha
		 i32 9, label %blahaha
		 i32 10, label %blahaha
	]

bb2:		; preds = %entry
	%tmp5 = and i32 %C, 123		; <i32> [#uses=1]
	%tmp = icmp eq i32 %tmp5, 0		; <i1> [#uses=1]
	br i1 %tmp, label %blahaha, label %cond_true

cond_true:		; preds = %bb2
	br label %blahaha

blahaha:		; preds = %cond_true, %bb2, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	%s.0 = phi i8* [ getelementptr ([8 x i8]* @str, i32 0, i64 0), %cond_true ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str1, i32 0, i64 0), %entry ], [ getelementptr ([5 x i8]* @str2, i32 0, i64 0), %bb2 ]		; <i8*> [#uses=13]
	%tmp8 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp10 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp12 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp14 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp16 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp18 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp20 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp22 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp24 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp26 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp28 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp30 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	%tmp32 = tail call i32 (i8*, ...)* @printf( i8* %s.0 )		; <i32> [#uses=0]
	ret void
}

declare i32 @printf(i8*, ...)
