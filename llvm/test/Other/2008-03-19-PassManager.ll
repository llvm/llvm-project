; PR 2034
; RUN: llvm-as < %s | opt -anders-aa -instcombine  -gvn -disable-output
	%struct.FULL = type { i32, i32, [1000 x float*] }

define i32 @sgesl(%struct.FULL* %a, i32* %ipvt, float* %b, i32 %job) {
entry:
	%a_addr = alloca %struct.FULL*		; <%struct.FULL**> [#uses=1]
	%ipvt_addr = alloca i32*		; <i32**> [#uses=1]
	%b_addr = alloca float*		; <float**> [#uses=1]
	%job_addr = alloca i32		; <i32*> [#uses=1]
	%akk = alloca float*		; <float**> [#uses=2]
	%k = alloca i32		; <i32*> [#uses=1]
	%l = alloca i32		; <i32*> [#uses=1]
	%n = alloca i32		; <i32*> [#uses=1]
	%nm1 = alloca i32		; <i32*> [#uses=1]
	%tmp5 = load i32* %job_addr, align 4		; <i32> [#uses=1]
	%tmp6 = icmp eq i32 %tmp5, 0		; <i1> [#uses=1]
	%tmp67 = zext i1 %tmp6 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp67, 0		; <i1> [#uses=1]
	br i1 %toBool, label %cond_true, label %cond_next137

cond_true:		; preds = %entry
	%tmp732 = load i32* %nm1, align 4		; <i32> [#uses=1]
	%tmp743 = icmp slt i32 0, %tmp732		; <i1> [#uses=1]
	%tmp74754 = zext i1 %tmp743 to i8		; <i8> [#uses=1]
	%toBool765 = icmp ne i8 %tmp74754, 0		; <i1> [#uses=1]
	br i1 %toBool765, label %bb, label %bb77

bb:		; preds = %cond_true
	%tmp9 = load %struct.FULL** %a_addr, align 4		; <%struct.FULL*> [#uses=1]
	%tmp10 = getelementptr %struct.FULL* %tmp9, i32 0, i32 2		; <[1000 x float*]*> [#uses=1]
	%tmp11 = getelementptr [1000 x float*]* %tmp10, i32 0, i32 0		; <float**> [#uses=1]
	%tmp12 = load float** %tmp11, align 4		; <float*> [#uses=1]
	%tmp13 = load i32* %k, align 4		; <i32> [#uses=1]
	%tmp14 = getelementptr float* %tmp12, i32 %tmp13		; <float*> [#uses=1]
	store float* %tmp14, float** %akk, align 4
	%tmp17 = load float** %b_addr, align 4		; <float*> [#uses=0]
	%tmp18 = load i32* %l, align 4		; <i32> [#uses=0]
	ret i32 0

bb77:		; preds = %cond_true
	ret i32 0

cond_next137:		; preds = %entry
	%tmp18922 = load i32* %n, align 4		; <i32> [#uses=1]
	%tmp19023 = icmp slt i32 0, %tmp18922		; <i1> [#uses=1]
	%tmp19019124 = zext i1 %tmp19023 to i8		; <i8> [#uses=1]
	%toBool19225 = icmp ne i8 %tmp19019124, 0		; <i1> [#uses=1]
	br i1 %toBool19225, label %bb138, label %bb193

bb138:		; preds = %cond_next137
	store float* null, float** %akk, align 4
	ret i32 0

bb193:		; preds = %cond_next137
	%tmp196 = load i32** %ipvt_addr, align 4		; <i32*> [#uses=0]
	ret i32 0
}
