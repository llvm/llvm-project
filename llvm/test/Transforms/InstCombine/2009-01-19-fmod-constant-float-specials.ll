; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -instcombine -S | grep 0x7FF8000000000000 | count 12
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -instcombine -S | grep "0\.0" | count 3
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -instcombine -S | grep "3\.5" | count 1
;

; ModuleID = 'apf.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
@"\01LC" = internal constant [4 x i8] c"%f\0A\00"		; <[4 x i8]*> [#uses=1]

define void @foo1() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0x7FF0000000000000, float* %x, align 4
	store float 0x7FF8000000000000, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare i32 @printf(i8*, ...) nounwind

define void @foo2() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0x7FF0000000000000, float* %x, align 4
	store float 0.000000e+00, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo3() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0x7FF0000000000000, float* %x, align 4
	store float 3.500000e+00, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo4() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0x7FF0000000000000, float* %x, align 4
	store float 0x7FF0000000000000, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo5() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0x7FF8000000000000, float* %x, align 4
	store float 0x7FF0000000000000, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo6() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0x7FF8000000000000, float* %x, align 4
	store float 0.000000e+00, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo7() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0x7FF8000000000000, float* %x, align 4
	store float 3.500000e+00, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo8() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0x7FF8000000000000, float* %x, align 4
	store float 0x7FF8000000000000, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo9() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0.000000e+00, float* %x, align 4
	store float 0x7FF8000000000000, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo10() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0.000000e+00, float* %x, align 4
	store float 0x7FF0000000000000, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo11() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0.000000e+00, float* %x, align 4
	store float 0.000000e+00, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo12() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 0.000000e+00, float* %x, align 4
	store float 3.500000e+00, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo13() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 3.500000e+00, float* %x, align 4
	store float 0x7FF8000000000000, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo14() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 3.500000e+00, float* %x, align 4
	store float 0x7FF0000000000000, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo15() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 3.500000e+00, float* %x, align 4
	store float 0.000000e+00, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

define void @foo16() nounwind {
entry:
	%y = alloca float		; <float*> [#uses=2]
	%x = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float 3.500000e+00, float* %x, align 4
	store float 3.500000e+00, float* %y, align 4
	%0 = load float, float* %y, align 4		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	%2 = load float, float* %x, align 4		; <float> [#uses=1]
	%3 = fpext float %2 to double		; <double> [#uses=1]
	%4 = frem double %3, %1		; <double> [#uses=1]
	%5 = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), double %4) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}
