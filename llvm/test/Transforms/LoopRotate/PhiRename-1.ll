; RUN: llvm-as < %s | opt -loop-rotate | llvm-dis | not grep {\\\[ .tmp224} 
; END.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"

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
@str1 = external constant [11 x i8]		; <[11 x i8]*> [#uses=1]
@operators = weak global %struct.list* null		; <%struct.list**> [#uses=1]



define i32 @opsOfArity(i32 %arity) {
entry:
	%arity_addr = alloca i32		; <i32*> [#uses=2]
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	%c = alloca i32, align 4		; <i32*> [#uses=4]
	%l = alloca %struct.list*, align 4		; <%struct.list**> [#uses=5]
	%op = alloca %struct.operator*, align 4		; <%struct.operator**> [#uses=3]
	"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %arity, i32* %arity_addr
	store i32 0, i32* %c
	%tmp1 = load %struct.list** @operators		; <%struct.list*> [#uses=1]
	store %struct.list* %tmp1, %struct.list** %l
	br label %bb21

bb:		; preds = %bb21
	%tmp3 = getelementptr %struct.list* %tmp22, i32 0, i32 0		; <i8**> [#uses=1]
	%tmp4 = load i8** %tmp3		; <i8*> [#uses=1]
	%tmp45 = bitcast i8* %tmp4 to %struct.operator*		; <%struct.operator*> [#uses=1]
	store %struct.operator* %tmp45, %struct.operator** %op
	%tmp6 = load %struct.operator** %op		; <%struct.operator*> [#uses=1]
	%tmp7 = getelementptr %struct.operator* %tmp6, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp8 = load i32* %tmp7		; <i32> [#uses=1]
	%tmp9 = load i32* %arity_addr		; <i32> [#uses=1]
	icmp eq i32 %tmp8, %tmp9		; <i1>:0 [#uses=1]
	zext i1 %0 to i8		; <i8>:1 [#uses=1]
	icmp ne i8 %1, 0		; <i1>:2 [#uses=1]
	br i1 %2, label %cond_true, label %cond_next

cond_true:		; preds = %bb
	%tmp10 = load %struct.operator** %op		; <%struct.operator*> [#uses=1]
	%tmp11 = getelementptr %struct.operator* %tmp10, i32 0, i32 2		; <i32*> [#uses=1]
	%tmp12 = load i32* %tmp11		; <i32> [#uses=1]
	%tmp13 = load %struct.FILE** @outfile		; <%struct.FILE*> [#uses=1]
	%tmp14 = getelementptr [11 x i8]* @str1, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp15 = call i32 (%struct.FILE*, i8*, ...)* @fprintf( %struct.FILE* %tmp13, i8* %tmp14, i32 %tmp12 )		; <i32> [#uses=0]
	%tmp16 = load i32* %c		; <i32> [#uses=1]
	%tmp17 = add i32 %tmp16, 1		; <i32> [#uses=1]
	store i32 %tmp17, i32* %c
	br label %cond_next

cond_next:		; preds = %cond_true, %bb
	%tmp19 = getelementptr %struct.list* %tmp22, i32 0, i32 1		; <%struct.list**> [#uses=1]
	%tmp20 = load %struct.list** %tmp19		; <%struct.list*> [#uses=1]
	store %struct.list* %tmp20, %struct.list** %l
	br label %bb21

bb21:		; preds = %cond_next, %entry
        %l.in = phi %struct.list** [ @operators, %entry ], [ %tmp19, %cond_next ]  ;
	%tmp22 = load %struct.list** %l.in		; <%struct.list*> [#uses=1]
	icmp ne %struct.list* %tmp22, null		; <i1>:3 [#uses=1]
	zext i1 %3 to i8		; <i8>:4 [#uses=1]
	icmp ne i8 %4, 0		; <i1>:5 [#uses=1]
	br i1 %5, label %bb, label %bb23

bb23:		; preds = %bb21
	%tmp24 = load i32* %c		; <i32> [#uses=1]
	store i32 %tmp24, i32* %tmp
	%tmp25 = load i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp25, i32* %retval
	br label %return

return:		; preds = %bb23
	%retval26 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval26
}

declare i32 @fprintf(%struct.FILE*, i8*, ...)
