; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--

define i128 @__fixsfdi(float %a) {
entry:
	%a_addr = alloca float		; <float*> [#uses=4]
	%retval = alloca i128, align 16		; <i128*> [#uses=2]
	%tmp = alloca i128, align 16		; <i128*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float %a, float* %a_addr
	%tmp1 = load float, float* %a_addr, align 4		; <float> [#uses=1]
	%tmp2 = fcmp olt float %tmp1, 0.000000e+00		; <i1> [#uses=1]
	%tmp23 = zext i1 %tmp2 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp23, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb, label %bb8
bb:		; preds = %entry
	%tmp4 = load float, float* %a_addr, align 4		; <float> [#uses=1]
	%tmp5 = fsub float -0.000000e+00, %tmp4		; <float> [#uses=1]
	%tmp6 = call i128 @__fixunssfDI( float %tmp5 ) nounwind 		; <i128> [#uses=1]
	%tmp7 = sub i128 0, %tmp6		; <i128> [#uses=1]
	store i128 %tmp7, i128* %tmp, align 16
	br label %bb11
bb8:		; preds = %entry
	%tmp9 = load float, float* %a_addr, align 4		; <float> [#uses=1]
	%tmp10 = call i128 @__fixunssfDI( float %tmp9 ) nounwind 		; <i128> [#uses=1]
	store i128 %tmp10, i128* %tmp, align 16
	br label %bb11
bb11:		; preds = %bb8, %bb
	%tmp12 = load i128, i128* %tmp, align 16		; <i128> [#uses=1]
	store i128 %tmp12, i128* %retval, align 16
	br label %return
return:		; preds = %bb11
	%retval13 = load i128, i128* %retval		; <i128> [#uses=1]
	ret i128 %retval13
}

declare i128 @__fixunssfDI(float)
