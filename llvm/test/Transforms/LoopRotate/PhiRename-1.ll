; RUN: opt < %s -passes=loop-rotate -verify-dom-info -verify-loop-info -verify-memoryssa -S | FileCheck %s
; CHECK-NOT: [ {{.}}tmp224

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"

	%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.Index_Map = type { i32, ptr }
	%struct.Item = type { [4 x i16], ptr }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
	%struct.dimension = type { ptr, %struct.Index_Map, ptr, i32, ptr }
	%struct.item_set = type { i32, i32, ptr, [2 x ptr], ptr, ptr, ptr, ptr }
	%struct.list = type { ptr, ptr }
	%struct.mapping = type { ptr, i32, i32, i32, ptr }
	%struct.nonterminal = type { ptr, i32, i32, i32, ptr, ptr }
	%struct.operator = type { ptr, i8, i32, i32, i32, i32, ptr }
	%struct.pattern = type { ptr, ptr, [2 x ptr] }
	%struct.plank = type { ptr, ptr, i32 }
	%struct.plankMap = type { ptr, i32, ptr }
	%struct.rule = type { [4 x i16], i32, i32, i32, ptr, ptr, i8 }
	%struct.stateMap = type { ptr, ptr, i32, ptr }
	%struct.table = type { ptr, ptr, ptr, [2 x ptr], ptr }
@outfile = external global ptr		; <ptr> [#uses=1]
@str1 = external constant [11 x i8]		; <ptr> [#uses=1]
@operators = weak global ptr null		; <ptr> [#uses=1]



define i32 @opsOfArity(i32 %arity) {
entry:
	%arity_addr = alloca i32		; <ptr> [#uses=2]
	%retval = alloca i32, align 4		; <ptr> [#uses=2]
	%tmp = alloca i32, align 4		; <ptr> [#uses=2]
	%c = alloca i32, align 4		; <ptr> [#uses=4]
	%l = alloca ptr, align 4		; <ptr> [#uses=5]
	%op = alloca ptr, align 4		; <ptr> [#uses=3]
	store i32 %arity, ptr %arity_addr
	store i32 0, ptr %c
	%tmp1 = load ptr, ptr @operators		; <ptr> [#uses=1]
	store ptr %tmp1, ptr %l
	br label %bb21

bb:		; preds = %bb21
	%tmp3 = getelementptr %struct.list, ptr %tmp22, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp4 = load ptr, ptr %tmp3		; <ptr> [#uses=1]
	store ptr %tmp4, ptr %op
	%tmp6 = load ptr, ptr %op		; <ptr> [#uses=1]
	%tmp7 = getelementptr %struct.operator, ptr %tmp6, i32 0, i32 5		; <ptr> [#uses=1]
	%tmp8 = load i32, ptr %tmp7		; <i32> [#uses=1]
	%tmp9 = load i32, ptr %arity_addr		; <i32> [#uses=1]
	icmp eq i32 %tmp8, %tmp9		; <i1>:0 [#uses=1]
	zext i1 %0 to i8		; <i8>:1 [#uses=1]
	icmp ne i8 %1, 0		; <i1>:2 [#uses=1]
	br i1 %2, label %cond_true, label %cond_next

cond_true:		; preds = %bb
	%tmp10 = load ptr, ptr %op		; <ptr> [#uses=1]
	%tmp11 = getelementptr %struct.operator, ptr %tmp10, i32 0, i32 2		; <ptr> [#uses=1]
	%tmp12 = load i32, ptr %tmp11		; <i32> [#uses=1]
	%tmp13 = load ptr, ptr @outfile		; <ptr> [#uses=1]
	%tmp14 = getelementptr [11 x i8], ptr @str1, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp15 = call i32 (ptr, ptr, ...) @fprintf( ptr %tmp13, ptr %tmp14, i32 %tmp12 )		; <i32> [#uses=0]
	%tmp16 = load i32, ptr %c		; <i32> [#uses=1]
	%tmp17 = add i32 %tmp16, 1		; <i32> [#uses=1]
	store i32 %tmp17, ptr %c
	br label %cond_next

cond_next:		; preds = %cond_true, %bb
	%tmp19 = getelementptr %struct.list, ptr %tmp22, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp20 = load ptr, ptr %tmp19		; <ptr> [#uses=1]
	store ptr %tmp20, ptr %l
	br label %bb21

bb21:		; preds = %cond_next, %entry
        %l.in = phi ptr [ @operators, %entry ], [ %tmp19, %cond_next ]
	%tmp22 = load ptr, ptr %l.in		; <ptr> [#uses=1]
	icmp ne ptr %tmp22, null		; <i1>:3 [#uses=1]
	zext i1 %3 to i8		; <i8>:4 [#uses=1]
	icmp ne i8 %4, 0		; <i1>:5 [#uses=1]
	br i1 %5, label %bb, label %bb23

bb23:		; preds = %bb21
	%tmp24 = load i32, ptr %c		; <i32> [#uses=1]
	store i32 %tmp24, ptr %tmp
	%tmp25 = load i32, ptr %tmp		; <i32> [#uses=1]
	store i32 %tmp25, ptr %retval
	br label %return

return:		; preds = %bb23
	%retval26 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval26
}

declare i32 @fprintf(ptr, ptr, ...)
