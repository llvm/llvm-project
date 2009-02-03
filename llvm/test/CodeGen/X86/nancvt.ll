; RUN: llvm-as < %s | opt -std-compile-opts | llc > %t
; RUN: grep 2147027116 %t | count 3
; RUN: grep 2147228864 %t | count 3
; RUN: grep 2146502828 %t | count 3
; RUN: grep 2143034560 %t | count 3
; Compile time conversions of NaNs.
; ModuleID = 'nan2.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
	%struct..0anon = type { float }
	%struct..1anon = type { double }
@fnan = constant [3 x i32] [ i32 2143831397, i32 2143831396, i32 2143831398 ]		; <[3 x i32]*> [#uses=1]
@dnan = constant [3 x i64] [ i64 9223235251041752696, i64 9223235251041752697, i64 9223235250773317239 ], align 8		; <[3 x i64]*> [#uses=1]
@fsnan = constant [3 x i32] [ i32 2139637093, i32 2139637092, i32 2139637094 ]		; <[3 x i32]*> [#uses=1]
@dsnan = constant [3 x i64] [ i64 9220983451228067448, i64 9220983451228067449, i64 9220983450959631991 ], align 8		; <[3 x i64]*> [#uses=1]
@.str = internal constant [10 x i8] c"%08x%08x\0A\00"		; <[10 x i8]*> [#uses=2]
@.str1 = internal constant [6 x i8] c"%08x\0A\00"		; <[6 x i8]*> [#uses=2]

define i32 @main() {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=1]
	%i = alloca i32, align 4		; <i32*> [#uses=20]
	%uf = alloca %struct..0anon, align 4		; <%struct..0anon*> [#uses=8]
	%ud = alloca %struct..1anon, align 8		; <%struct..1anon*> [#uses=10]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 0, i32* %i, align 4
	br label %bb23

bb:		; preds = %bb23
	%tmp = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp1 = getelementptr [3 x i32]* @fnan, i32 0, i32 %tmp		; <i32*> [#uses=1]
	%tmp2 = load i32* %tmp1, align 4		; <i32> [#uses=1]
	%tmp3 = getelementptr %struct..0anon* %uf, i32 0, i32 0		; <float*> [#uses=1]
	%tmp34 = bitcast float* %tmp3 to i32*		; <i32*> [#uses=1]
	store i32 %tmp2, i32* %tmp34, align 4
	%tmp5 = getelementptr %struct..0anon* %uf, i32 0, i32 0		; <float*> [#uses=1]
	%tmp6 = load float* %tmp5, align 4		; <float> [#uses=1]
	%tmp67 = fpext float %tmp6 to double		; <double> [#uses=1]
	%tmp8 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	store double %tmp67, double* %tmp8, align 8
	%tmp9 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	%tmp910 = bitcast double* %tmp9 to i64*		; <i64*> [#uses=1]
	%tmp11 = load i64* %tmp910, align 8		; <i64> [#uses=1]
	%tmp1112 = trunc i64 %tmp11 to i32		; <i32> [#uses=1]
	%tmp13 = and i32 %tmp1112, -1		; <i32> [#uses=1]
	%tmp14 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	%tmp1415 = bitcast double* %tmp14 to i64*		; <i64*> [#uses=1]
	%tmp16 = load i64* %tmp1415, align 8		; <i64> [#uses=1]
	%.cast = zext i32 32 to i64		; <i64> [#uses=1]
	%tmp17 = ashr i64 %tmp16, %.cast		; <i64> [#uses=1]
	%tmp1718 = trunc i64 %tmp17 to i32		; <i32> [#uses=1]
	%tmp19 = getelementptr [10 x i8]* @.str, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp20 = call i32 (i8*, ...)* @printf( i8* %tmp19, i32 %tmp1718, i32 %tmp13 )		; <i32> [#uses=0]
	%tmp21 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp22 = add i32 %tmp21, 1		; <i32> [#uses=1]
	store i32 %tmp22, i32* %i, align 4
	br label %bb23

bb23:		; preds = %bb, %entry
	%tmp24 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp25 = icmp sle i32 %tmp24, 2		; <i1> [#uses=1]
	%tmp2526 = zext i1 %tmp25 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp2526, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb, label %bb27

bb27:		; preds = %bb23
	store i32 0, i32* %i, align 4
	br label %bb46

bb28:		; preds = %bb46
	%tmp29 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp30 = getelementptr [3 x i64]* @dnan, i32 0, i32 %tmp29		; <i64*> [#uses=1]
	%tmp31 = load i64* %tmp30, align 8		; <i64> [#uses=1]
	%tmp32 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	%tmp3233 = bitcast double* %tmp32 to i64*		; <i64*> [#uses=1]
	store i64 %tmp31, i64* %tmp3233, align 8
	%tmp35 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	%tmp36 = load double* %tmp35, align 8		; <double> [#uses=1]
	%tmp3637 = fptrunc double %tmp36 to float		; <float> [#uses=1]
	%tmp38 = getelementptr %struct..0anon* %uf, i32 0, i32 0		; <float*> [#uses=1]
	store float %tmp3637, float* %tmp38, align 4
	%tmp39 = getelementptr %struct..0anon* %uf, i32 0, i32 0		; <float*> [#uses=1]
	%tmp3940 = bitcast float* %tmp39 to i32*		; <i32*> [#uses=1]
	%tmp41 = load i32* %tmp3940, align 4		; <i32> [#uses=1]
	%tmp42 = getelementptr [6 x i8]* @.str1, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp43 = call i32 (i8*, ...)* @printf( i8* %tmp42, i32 %tmp41 )		; <i32> [#uses=0]
	%tmp44 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp45 = add i32 %tmp44, 1		; <i32> [#uses=1]
	store i32 %tmp45, i32* %i, align 4
	br label %bb46

bb46:		; preds = %bb28, %bb27
	%tmp47 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp48 = icmp sle i32 %tmp47, 2		; <i1> [#uses=1]
	%tmp4849 = zext i1 %tmp48 to i8		; <i8> [#uses=1]
	%toBool50 = icmp ne i8 %tmp4849, 0		; <i1> [#uses=1]
	br i1 %toBool50, label %bb28, label %bb51

bb51:		; preds = %bb46
	store i32 0, i32* %i, align 4
	br label %bb78

bb52:		; preds = %bb78
	%tmp53 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp54 = getelementptr [3 x i32]* @fsnan, i32 0, i32 %tmp53		; <i32*> [#uses=1]
	%tmp55 = load i32* %tmp54, align 4		; <i32> [#uses=1]
	%tmp56 = getelementptr %struct..0anon* %uf, i32 0, i32 0		; <float*> [#uses=1]
	%tmp5657 = bitcast float* %tmp56 to i32*		; <i32*> [#uses=1]
	store i32 %tmp55, i32* %tmp5657, align 4
	%tmp58 = getelementptr %struct..0anon* %uf, i32 0, i32 0		; <float*> [#uses=1]
	%tmp59 = load float* %tmp58, align 4		; <float> [#uses=1]
	%tmp5960 = fpext float %tmp59 to double		; <double> [#uses=1]
	%tmp61 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	store double %tmp5960, double* %tmp61, align 8
	%tmp62 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	%tmp6263 = bitcast double* %tmp62 to i64*		; <i64*> [#uses=1]
	%tmp64 = load i64* %tmp6263, align 8		; <i64> [#uses=1]
	%tmp6465 = trunc i64 %tmp64 to i32		; <i32> [#uses=1]
	%tmp66 = and i32 %tmp6465, -1		; <i32> [#uses=1]
	%tmp68 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	%tmp6869 = bitcast double* %tmp68 to i64*		; <i64*> [#uses=1]
	%tmp70 = load i64* %tmp6869, align 8		; <i64> [#uses=1]
	%.cast71 = zext i32 32 to i64		; <i64> [#uses=1]
	%tmp72 = ashr i64 %tmp70, %.cast71		; <i64> [#uses=1]
	%tmp7273 = trunc i64 %tmp72 to i32		; <i32> [#uses=1]
	%tmp74 = getelementptr [10 x i8]* @.str, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp75 = call i32 (i8*, ...)* @printf( i8* %tmp74, i32 %tmp7273, i32 %tmp66 )		; <i32> [#uses=0]
	%tmp76 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp77 = add i32 %tmp76, 1		; <i32> [#uses=1]
	store i32 %tmp77, i32* %i, align 4
	br label %bb78

bb78:		; preds = %bb52, %bb51
	%tmp79 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp80 = icmp sle i32 %tmp79, 2		; <i1> [#uses=1]
	%tmp8081 = zext i1 %tmp80 to i8		; <i8> [#uses=1]
	%toBool82 = icmp ne i8 %tmp8081, 0		; <i1> [#uses=1]
	br i1 %toBool82, label %bb52, label %bb83

bb83:		; preds = %bb78
	store i32 0, i32* %i, align 4
	br label %bb101

bb84:		; preds = %bb101
	%tmp85 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp86 = getelementptr [3 x i64]* @dsnan, i32 0, i32 %tmp85		; <i64*> [#uses=1]
	%tmp87 = load i64* %tmp86, align 8		; <i64> [#uses=1]
	%tmp88 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	%tmp8889 = bitcast double* %tmp88 to i64*		; <i64*> [#uses=1]
	store i64 %tmp87, i64* %tmp8889, align 8
	%tmp90 = getelementptr %struct..1anon* %ud, i32 0, i32 0		; <double*> [#uses=1]
	%tmp91 = load double* %tmp90, align 8		; <double> [#uses=1]
	%tmp9192 = fptrunc double %tmp91 to float		; <float> [#uses=1]
	%tmp93 = getelementptr %struct..0anon* %uf, i32 0, i32 0		; <float*> [#uses=1]
	store float %tmp9192, float* %tmp93, align 4
	%tmp94 = getelementptr %struct..0anon* %uf, i32 0, i32 0		; <float*> [#uses=1]
	%tmp9495 = bitcast float* %tmp94 to i32*		; <i32*> [#uses=1]
	%tmp96 = load i32* %tmp9495, align 4		; <i32> [#uses=1]
	%tmp97 = getelementptr [6 x i8]* @.str1, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp98 = call i32 (i8*, ...)* @printf( i8* %tmp97, i32 %tmp96 )		; <i32> [#uses=0]
	%tmp99 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp100 = add i32 %tmp99, 1		; <i32> [#uses=1]
	store i32 %tmp100, i32* %i, align 4
	br label %bb101

bb101:		; preds = %bb84, %bb83
	%tmp102 = load i32* %i, align 4		; <i32> [#uses=1]
	%tmp103 = icmp sle i32 %tmp102, 2		; <i1> [#uses=1]
	%tmp103104 = zext i1 %tmp103 to i8		; <i8> [#uses=1]
	%toBool105 = icmp ne i8 %tmp103104, 0		; <i1> [#uses=1]
	br i1 %toBool105, label %bb84, label %bb106

bb106:		; preds = %bb101
	br label %return

return:		; preds = %bb106
	%retval107 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval107
}

declare i32 @printf(i8*, ...)
