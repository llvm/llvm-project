; Make sure this doesn't turn into an infinite loop

; RUN: opt < %s -passes=simplifycfg,instsimplify,simplifycfg -simplifycfg-require-and-preserve-domtree=1 | llvm-dis | FileCheck %s

%struct.anon = type { i32, i32, i32, i32, [1024 x i8] }
@_zero_ = external global ptr		; <ptr> [#uses=2]
@_one_ = external global ptr		; <ptr> [#uses=4]
@str = internal constant [4 x i8] c"%d\0A\00"		; <ptr> [#uses=0]

declare i32 @bc_compare(ptr, ptr)

declare void @free_num(ptr)

declare ptr @copy_num(ptr)

declare void @init_num(ptr)

declare ptr @new_num(i32, i32)

declare void @int2num(ptr, i32)

declare void @bc_multiply(ptr, ptr, ptr, i32)

declare void @bc_raise(ptr, ptr, ptr, i32)

declare i32 @bc_divide(ptr, ptr, ptr, i32)

declare void @bc_add(ptr, ptr, ptr)

declare i32 @_do_compare(ptr, ptr, i32, i32)

declare i32 @printf(ptr, ...)

define i32 @bc_sqrt(ptr %num, i32 %scale) {
entry:
	%guess = alloca ptr		; <ptr> [#uses=7]
	%guess1 = alloca ptr		; <ptr> [#uses=7]
	%point5 = alloca ptr		; <ptr> [#uses=3]
	%tmp = load ptr, ptr %num		; <ptr> [#uses=1]
	%tmp1 = load ptr, ptr @_zero_		; <ptr> [#uses=1]
	%tmp.upgrd.1 = call i32 @bc_compare( ptr %tmp, ptr %tmp1 )		; <i32> [#uses=2]
	%tmp.upgrd.2 = icmp slt i32 %tmp.upgrd.1, 0		; <i1> [#uses=1]
	br i1 %tmp.upgrd.2, label %cond_true, label %cond_false
cond_true:		; preds = %entry
	ret i32 0
cond_false:		; preds = %entry
	%tmp5 = icmp eq i32 %tmp.upgrd.1, 0		; <i1> [#uses=1]
	br i1 %tmp5, label %cond_true6, label %cond_next13
cond_true6:		; preds = %cond_false
	call void @free_num( ptr %num )
	%tmp8 = load ptr, ptr @_zero_		; <ptr> [#uses=1]
	%tmp9 = call ptr @copy_num( ptr %tmp8 )		; <ptr> [#uses=1]
	store ptr %tmp9, ptr %num
	ret i32 1
cond_next13:		; preds = %cond_false
	%tmp15 = load ptr, ptr %num		; <ptr> [#uses=1]
	%tmp16 = load ptr, ptr @_one_		; <ptr> [#uses=1]
	%tmp17 = call i32 @bc_compare( ptr %tmp15, ptr %tmp16 )		; <i32> [#uses=2]
	%tmp19 = icmp eq i32 %tmp17, 0		; <i1> [#uses=1]
	br i1 %tmp19, label %cond_true20, label %cond_next27
cond_true20:		; preds = %cond_next13
	call void @free_num( ptr %num )
	%tmp22 = load ptr, ptr @_one_		; <ptr> [#uses=1]
	%tmp23 = call ptr @copy_num( ptr %tmp22 )		; <ptr> [#uses=1]
	store ptr %tmp23, ptr %num
	ret i32 1
cond_next27:		; preds = %cond_next13
	%tmp29 = load ptr, ptr %num		; <ptr> [#uses=1]
	%tmp30 = getelementptr %struct.anon, ptr %tmp29, i32 0, i32 2		; <ptr> [#uses=1]
	%tmp31 = load i32, ptr %tmp30		; <i32> [#uses=2]
	%tmp33 = icmp sge i32 %tmp31, %scale		; <i1> [#uses=1]
	%max = select i1 %tmp33, i32 %tmp31, i32 %scale		; <i32> [#uses=4]
	%tmp35 = add i32 %max, 2		; <i32> [#uses=0]
	call void @init_num( ptr %guess )
	call void @init_num( ptr %guess1 )
	%tmp36 = call ptr @new_num( i32 1, i32 1 )		; <ptr> [#uses=2]
	store ptr %tmp36, ptr %point5
	%tmp.upgrd.3 = getelementptr %struct.anon, ptr %tmp36, i32 0, i32 4, i32 1		; <ptr> [#uses=1]
	store i8 5, ptr %tmp.upgrd.3
	%tmp39 = icmp slt i32 %tmp17, 0		; <i1> [#uses=1]
	br i1 %tmp39, label %cond_true40, label %cond_false43
cond_true40:		; preds = %cond_next27
	%tmp41 = load ptr, ptr @_one_		; <ptr> [#uses=1]
	%tmp42 = call ptr @copy_num( ptr %tmp41 )		; <ptr> [#uses=1]
	store ptr %tmp42, ptr %guess
	br label %bb80.outer
cond_false43:		; preds = %cond_next27
	call void @int2num( ptr %guess, i32 10 )
	%tmp45 = load ptr, ptr %num		; <ptr> [#uses=1]
	%tmp46 = getelementptr %struct.anon, ptr %tmp45, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp47 = load i32, ptr %tmp46		; <i32> [#uses=1]
	call void @int2num( ptr %guess1, i32 %tmp47 )
	%tmp48 = load ptr, ptr %guess1		; <ptr> [#uses=1]
	%tmp49 = load ptr, ptr %point5		; <ptr> [#uses=1]
	call void @bc_multiply( ptr %tmp48, ptr %tmp49, ptr %guess1, i32 %max )
	%tmp51 = load ptr, ptr %guess1		; <ptr> [#uses=1]
	%tmp52 = getelementptr %struct.anon, ptr %tmp51, i32 0, i32 2		; <ptr> [#uses=1]
	store i32 0, ptr %tmp52
	%tmp53 = load ptr, ptr %guess		; <ptr> [#uses=1]
	%tmp54 = load ptr, ptr %guess1		; <ptr> [#uses=1]
	call void @bc_raise( ptr %tmp53, ptr %tmp54, ptr %guess, i32 %max )
	br label %bb80.outer
bb80.outer:		; preds = %cond_true83, %cond_false43, %cond_true40
	%done.1.ph = phi i32 [ 1, %cond_true83 ], [ 0, %cond_true40 ], [ 0, %cond_false43 ]		; <i32> [#uses=1]
	br label %bb80
bb80:		; preds = %cond_true83, %bb80.outer
	%tmp82 = icmp eq i32 %done.1.ph, 0		; <i1> [#uses=1]
	br i1 %tmp82, label %cond_true83, label %bb86
cond_true83:		; preds = %bb80
	%tmp71 = call i32 @_do_compare( ptr null, ptr null, i32 0, i32 1 )		; <i32> [#uses=1]
	%tmp76 = icmp eq i32 %tmp71, 0		; <i1> [#uses=1]
	br i1 %tmp76, label %bb80.outer, label %bb80
; CHECK: bb86
bb86:		; preds = %bb80
	call void @free_num( ptr %num )
	%tmp88 = load ptr, ptr %guess		; <ptr> [#uses=1]
	%tmp89 = load ptr, ptr @_one_		; <ptr> [#uses=1]
	%tmp92 = call i32 @bc_divide( ptr %tmp88, ptr %tmp89, ptr %num, i32 %max )		; <i32> [#uses=0]
	call void @free_num( ptr %guess )
	call void @free_num( ptr %guess1 )
	call void @free_num( ptr %point5 )
	ret i32 1
}
