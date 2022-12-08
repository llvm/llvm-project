; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 | llvm-dis
; END.

; ModuleID = '2006-12-08-Ptr-ICmp-Branch.ll'
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, i32 }
	%struct.charsequence = type { i8*, i32, i32 }
	%struct.trie_s = type { [26 x %struct.trie_s*], i32 }
@str = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@str.upgrd.1 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@str.upgrd.2 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@C.0.2294 = external global %struct.charsequence		; <%struct.charsequence*> [#uses=3]
@t = external global %struct.trie_s*		; <%struct.trie_s**> [#uses=0]
@str.upgrd.3 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@str.upgrd.4 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]

declare void @charsequence_reset(%struct.charsequence*)

declare void @free(i8*)

declare void @charsequence_push(%struct.charsequence*, i8)

declare i8* @charsequence_val(%struct.charsequence*)

declare i32 @_IO_getc(%struct.FILE*)

declare i32 @tolower(i32)

declare %struct.trie_s* @trie_insert(%struct.trie_s*, i8*)

declare i32 @feof(%struct.FILE*)

define void @addfile(%struct.trie_s* %t, %struct.FILE* %f) {
entry:
	%t_addr = alloca %struct.trie_s*		; <%struct.trie_s**> [#uses=2]
	%f_addr = alloca %struct.FILE*		; <%struct.FILE**> [#uses=3]
	%c = alloca i8, align 1		; <i8*> [#uses=7]
	%wstate = alloca i32, align 4		; <i32*> [#uses=4]
	%cs = alloca %struct.charsequence, align 16		; <%struct.charsequence*> [#uses=7]
	%str = alloca i8*, align 4		; <i8**> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store %struct.trie_s* %t, %struct.trie_s** %t_addr
	store %struct.FILE* %f, %struct.FILE** %f_addr
	store i32 0, i32* %wstate
	%tmp = getelementptr %struct.charsequence, %struct.charsequence* %cs, i64 0, i32 0		; <i8**> [#uses=1]
	%tmp1 = getelementptr %struct.charsequence, %struct.charsequence* @C.0.2294, i64 0, i32 0		; <i8**> [#uses=1]
	%tmp.upgrd.5 = load i8*, i8** %tmp1		; <i8*> [#uses=1]
	store i8* %tmp.upgrd.5, i8** %tmp
	%tmp.upgrd.6 = getelementptr %struct.charsequence, %struct.charsequence* %cs, i64 0, i32 1		; <i32*> [#uses=1]
	%tmp2 = getelementptr %struct.charsequence, %struct.charsequence* @C.0.2294, i64 0, i32 1		; <i32*> [#uses=1]
	%tmp.upgrd.7 = load i32, i32* %tmp2		; <i32> [#uses=1]
	store i32 %tmp.upgrd.7, i32* %tmp.upgrd.6
	%tmp3 = getelementptr %struct.charsequence, %struct.charsequence* %cs, i64 0, i32 2		; <i32*> [#uses=1]
	%tmp4 = getelementptr %struct.charsequence, %struct.charsequence* @C.0.2294, i64 0, i32 2		; <i32*> [#uses=1]
	%tmp5 = load i32, i32* %tmp4		; <i32> [#uses=1]
	store i32 %tmp5, i32* %tmp3
	br label %bb33
bb:		; preds = %bb33
	%tmp.upgrd.8 = load %struct.FILE*, %struct.FILE** %f_addr		; <%struct.FILE*> [#uses=1]
	%tmp.upgrd.9 = call i32 @_IO_getc( %struct.FILE* %tmp.upgrd.8 )		; <i32> [#uses=1]
	%tmp6 = call i32 @tolower( i32 %tmp.upgrd.9 )		; <i32> [#uses=1]
	%tmp6.upgrd.10 = trunc i32 %tmp6 to i8		; <i8> [#uses=1]
	store i8 %tmp6.upgrd.10, i8* %c
	%tmp7 = load i32, i32* %wstate		; <i32> [#uses=1]
	%tmp.upgrd.11 = icmp ne i32 %tmp7, 0		; <i1> [#uses=1]
	br i1 %tmp.upgrd.11, label %cond_true, label %cond_false
cond_true:		; preds = %bb
	%tmp.upgrd.12 = load i8, i8* %c		; <i8> [#uses=1]
	%tmp8 = icmp sle i8 %tmp.upgrd.12, 96		; <i1> [#uses=1]
	br i1 %tmp8, label %cond_true9, label %cond_next
cond_true9:		; preds = %cond_true
	br label %bb16
cond_next:		; preds = %cond_true
	%tmp10 = load i8, i8* %c		; <i8> [#uses=1]
	%tmp11 = icmp sgt i8 %tmp10, 122		; <i1> [#uses=1]
	br i1 %tmp11, label %cond_true12, label %cond_next13
cond_true12:		; preds = %cond_next
	br label %bb16
cond_next13:		; preds = %cond_next
	%tmp14 = load i8, i8* %c		; <i8> [#uses=1]
	%tmp14.upgrd.13 = sext i8 %tmp14 to i32		; <i32> [#uses=1]
	%tmp1415 = trunc i32 %tmp14.upgrd.13 to i8		; <i8> [#uses=1]
	call void @charsequence_push( %struct.charsequence* %cs, i8 %tmp1415 )
	br label %bb21
bb16:		; preds = %cond_true12, %cond_true9
	%tmp17 = call i8* @charsequence_val( %struct.charsequence* %cs )		; <i8*> [#uses=1]
	store i8* %tmp17, i8** %str
	%tmp.upgrd.14 = load %struct.trie_s*, %struct.trie_s** %t_addr		; <%struct.trie_s*> [#uses=1]
	%tmp18 = load i8*, i8** %str		; <i8*> [#uses=1]
	%tmp19 = call %struct.trie_s* @trie_insert( %struct.trie_s* %tmp.upgrd.14, i8* %tmp18 )		; <%struct.trie_s*> [#uses=0]
	%tmp20 = load i8*, i8** %str		; <i8*> [#uses=1]
	call void @free( i8* %tmp20 )
	store i32 0, i32* %wstate
	br label %bb21
bb21:		; preds = %bb16, %cond_next13
	br label %cond_next32
cond_false:		; preds = %bb
	%tmp22 = load i8, i8* %c		; <i8> [#uses=1]
	%tmp23 = icmp sgt i8 %tmp22, 96		; <i1> [#uses=1]
	br i1 %tmp23, label %cond_true24, label %cond_next31
cond_true24:		; preds = %cond_false
	%tmp25 = load i8, i8* %c		; <i8> [#uses=1]
	%tmp26 = icmp sle i8 %tmp25, 122		; <i1> [#uses=1]
	br i1 %tmp26, label %cond_true27, label %cond_next30
cond_true27:		; preds = %cond_true24
	call void @charsequence_reset( %struct.charsequence* %cs )
	%tmp28 = load i8, i8* %c		; <i8> [#uses=1]
	%tmp28.upgrd.15 = sext i8 %tmp28 to i32		; <i32> [#uses=1]
	%tmp2829 = trunc i32 %tmp28.upgrd.15 to i8		; <i8> [#uses=1]
	call void @charsequence_push( %struct.charsequence* %cs, i8 %tmp2829 )
	store i32 1, i32* %wstate
	br label %cond_next30
cond_next30:		; preds = %cond_true27, %cond_true24
	br label %cond_next31
cond_next31:		; preds = %cond_next30, %cond_false
	br label %cond_next32
cond_next32:		; preds = %cond_next31, %bb21
	br label %bb33
bb33:		; preds = %cond_next32, %entry
	%tmp34 = load %struct.FILE*, %struct.FILE** %f_addr		; <%struct.FILE*> [#uses=1]
	%tmp35 = call i32 @feof( %struct.FILE* %tmp34 )		; <i32> [#uses=1]
	%tmp36 = icmp eq i32 %tmp35, 0		; <i1> [#uses=1]
	br i1 %tmp36, label %bb, label %bb37
bb37:		; preds = %bb33
	br label %return
return:		; preds = %bb37
	ret void
}
