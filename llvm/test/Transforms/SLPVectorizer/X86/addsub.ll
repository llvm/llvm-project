; RUN: opt < %s -basicaa -slp-vectorizer -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = common global [4 x i32] zeroinitializer, align 16
@c = common global [4 x i32] zeroinitializer, align 16
@d = common global [4 x i32] zeroinitializer, align 16
@e = common global [4 x i32] zeroinitializer, align 16
@a = common global [4 x i32] zeroinitializer, align 16
@fb = common global [4 x float] zeroinitializer, align 16
@fc = common global [4 x float] zeroinitializer, align 16
@fa = common global [4 x float] zeroinitializer, align 16
@fd = common global [4 x float] zeroinitializer, align 16

; CHECK-LABEL: @addsub
; CHECK: %5 = add nsw <4 x i32> %3, %4
; CHECK: %6 = add nsw <4 x i32> %2, %5
; CHECK: %7 = sub nsw <4 x i32> %2, %5
; CHECK: %8 = shufflevector <4 x i32> %6, <4 x i32> %7, <4 x i32> <i32 0, i32 5, i32 2, i32 7>

; Function Attrs: nounwind uwtable
define void @addsub() #0 {
entry:
  %0 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i64 0), align 4
  %1 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i64 0), align 4
  %add = add nsw i32 %0, %1
  %2 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @d, i32 0, i64 0), align 4
  %3 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @e, i32 0, i64 0), align 4
  %add1 = add nsw i32 %2, %3
  %add2 = add nsw i32 %add, %add1
  store i32 %add2, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i32 0, i64 0), align 4
  %4 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i64 1), align 4
  %5 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i64 1), align 4
  %add3 = add nsw i32 %4, %5
  %6 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @d, i32 0, i64 1), align 4
  %7 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @e, i32 0, i64 1), align 4
  %add4 = add nsw i32 %6, %7
  %sub = sub nsw i32 %add3, %add4
  store i32 %sub, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i32 0, i64 1), align 4
  %8 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i64 2), align 4
  %9 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i64 2), align 4
  %add5 = add nsw i32 %8, %9
  %10 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @d, i32 0, i64 2), align 4
  %11 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @e, i32 0, i64 2), align 4
  %add6 = add nsw i32 %10, %11
  %add7 = add nsw i32 %add5, %add6
  store i32 %add7, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i32 0, i64 2), align 4
  %12 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i64 3), align 4
  %13 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i64 3), align 4
  %add8 = add nsw i32 %12, %13
  %14 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @d, i32 0, i64 3), align 4
  %15 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @e, i32 0, i64 3), align 4
  %add9 = add nsw i32 %14, %15
  %sub10 = sub nsw i32 %add8, %add9
  store i32 %sub10, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i32 0, i64 3), align 4
  ret void
}

; CHECK-LABEL: @subadd
; CHECK:  %5 = add nsw <4 x i32> %3, %4
; CHECK:  %6 = sub nsw <4 x i32> %2, %5
; CHECK:  %7 = add nsw <4 x i32> %2, %5
; CHECK:  %8 = shufflevector <4 x i32> %6, <4 x i32> %7, <4 x i32> <i32 0, i32 5, i32 2, i32 7>

; Function Attrs: nounwind uwtable
define void @subadd() #0 {
entry:
  %0 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i64 0), align 4
  %1 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i64 0), align 4
  %add = add nsw i32 %0, %1
  %2 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @d, i32 0, i64 0), align 4
  %3 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @e, i32 0, i64 0), align 4
  %add1 = add nsw i32 %2, %3
  %sub = sub nsw i32 %add, %add1
  store i32 %sub, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i32 0, i64 0), align 4
  %4 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i64 1), align 4
  %5 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i64 1), align 4
  %add2 = add nsw i32 %4, %5
  %6 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @d, i32 0, i64 1), align 4
  %7 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @e, i32 0, i64 1), align 4
  %add3 = add nsw i32 %6, %7
  %add4 = add nsw i32 %add2, %add3
  store i32 %add4, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i32 0, i64 1), align 4
  %8 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i64 2), align 4
  %9 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i64 2), align 4
  %add5 = add nsw i32 %8, %9
  %10 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @d, i32 0, i64 2), align 4
  %11 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @e, i32 0, i64 2), align 4
  %add6 = add nsw i32 %10, %11
  %sub7 = sub nsw i32 %add5, %add6
  store i32 %sub7, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i32 0, i64 2), align 4
  %12 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i64 3), align 4
  %13 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i64 3), align 4
  %add8 = add nsw i32 %12, %13
  %14 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @d, i32 0, i64 3), align 4
  %15 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @e, i32 0, i64 3), align 4
  %add9 = add nsw i32 %14, %15
  %add10 = add nsw i32 %add8, %add9
  store i32 %add10, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i32 0, i64 3), align 4
  ret void
}

; CHECK-LABEL: @faddfsub
; CHECK: %2 = fadd <4 x float> %0, %1
; CHECK: %3 = fsub <4 x float> %0, %1
; CHECK: %4 = shufflevector <4 x float> %2, <4 x float> %3, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; Function Attrs: nounwind uwtable
define void @faddfsub() #0 {
entry:
  %0 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 0), align 4
  %1 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 0), align 4
  %add = fadd float %0, %1
  store float %add, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 0), align 4
  %2 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 1), align 4
  %3 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 1), align 4
  %sub = fsub float %2, %3
  store float %sub, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 1), align 4
  %4 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 2), align 4
  %5 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 2), align 4
  %add1 = fadd float %4, %5
  store float %add1, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 2), align 4
  %6 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 3), align 4
  %7 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 3), align 4
  %sub2 = fsub float %6, %7
  store float %sub2, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 3), align 4
  ret void
}

; CHECK-LABEL: @fsubfadd
; CHECK: %2 = fsub <4 x float> %0, %1
; CHECK: %3 = fadd <4 x float> %0, %1
; CHECK: %4 = shufflevector <4 x float> %2, <4 x float> %3, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; Function Attrs: nounwind uwtable
define void @fsubfadd() #0 {
entry:
  %0 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 0), align 4
  %1 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 0), align 4
  %sub = fsub float %0, %1
  store float %sub, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 0), align 4
  %2 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 1), align 4
  %3 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 1), align 4
  %add = fadd float %2, %3
  store float %add, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 1), align 4
  %4 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 2), align 4
  %5 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 2), align 4
  %sub1 = fsub float %4, %5
  store float %sub1, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 2), align 4
  %6 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 3), align 4
  %7 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 3), align 4
  %add2 = fadd float %6, %7
  store float %add2, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 3), align 4
  ret void
}

; CHECK-LABEL: @No_faddfsub
; CHECK-NOT: fadd <4 x float>
; CHECK-NOT: fsub <4 x float>
; CHECK-NOT: shufflevector
; Function Attrs: nounwind uwtable
define void @No_faddfsub() #0 {
entry:
  %0 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 0), align 4
  %1 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 0), align 4
  %add = fadd float %0, %1
  store float %add, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 0), align 4
  %2 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 1), align 4
  %3 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 1), align 4
  %add1 = fadd float %2, %3
  store float %add1, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 1), align 4
  %4 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 2), align 4
  %5 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 2), align 4
  %add2 = fadd float %4, %5
  store float %add2, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 2), align 4
  %6 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 3), align 4
  %7 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 3), align 4
  %sub = fsub float %6, %7
  store float %sub, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 3), align 4
  ret void
}

; Check vectorization of following code for float data type-
;  fc[0] = fb[0]+fa[0]; //swapped fb and fa
;  fc[1] = fa[1]-fb[1];
;  fc[2] = fa[2]+fb[2];
;  fc[3] = fa[3]-fb[3];

; CHECK-LABEL: @reorder_alt
; CHECK: %3 = fadd <4 x float> %1, %2
; CHECK: %4 = fsub <4 x float> %1, %2
; CHECK: %5 = shufflevector <4 x float> %3, <4 x float> %4, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
define void @reorder_alt() #0 {
  %1 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 0), align 4
  %2 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 0), align 4
  %3 = fadd float %1, %2
  store float %3, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 0), align 4
  %4 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 1), align 4
  %5 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 1), align 4
  %6 = fsub float %4, %5
  store float %6, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 1), align 4
  %7 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 2), align 4
  %8 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 2), align 4
  %9 = fadd float %7, %8
  store float %9, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 2), align 4
  %10 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 3), align 4
  %11 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 3), align 4
  %12 = fsub float %10, %11
  store float %12, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 3), align 4
  ret void
}

; Check vectorization of following code for float data type-
;  fc[0] = fa[0]+(fb[0]-fd[0]);
;  fc[1] = fa[1]-(fb[1]+fd[1]);
;  fc[2] = fa[2]+(fb[2]-fd[2]);
;  fc[3] = fa[3]-(fd[3]+fb[3]); //swapped fd and fb 

; CHECK-LABEL: @reorder_alt_subTree
; CHECK: %4 = fsub <4 x float> %3, %2
; CHECK: %5 = fadd <4 x float> %3, %2
; CHECK: %6 = shufflevector <4 x float> %4, <4 x float> %5, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; CHECK: %7 = fadd <4 x float> %1, %6
; CHECK: %8 = fsub <4 x float> %1, %6
; CHECK: %9 = shufflevector <4 x float> %7, <4 x float> %8, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
define void @reorder_alt_subTree() #0 {
  %1 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 0), align 4
  %2 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 0), align 4
  %3 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fd, i32 0, i64 0), align 4
  %4 = fsub float %2, %3
  %5 = fadd float %1, %4
  store float %5, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 0), align 4
  %6 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 1), align 4
  %7 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 1), align 4
  %8 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fd, i32 0, i64 1), align 4
  %9 = fadd float %7, %8
  %10 = fsub float %6, %9
  store float %10, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 1), align 4
  %11 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 2), align 4
  %12 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 2), align 4
  %13 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fd, i32 0, i64 2), align 4
  %14 = fsub float %12, %13
  %15 = fadd float %11, %14
  store float %15, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 2), align 4
  %16 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 3), align 4
  %17 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fd, i32 0, i64 3), align 4
  %18 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 3), align 4
  %19 = fadd float %17, %18
  %20 = fsub float %16, %19
  store float %20, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 3), align 4
  ret void
}

; Check vectorization of following code for double data type-
;  c[0] = (a[0]+b[0])-d[0];
;  c[1] = d[1]+(a[1]+b[1]); //swapped d[1] and (a[1]+b[1]) 

; CHECK-LABEL: @reorder_alt_rightsubTree
; CHECK: fadd <2 x double>
; CHECK: fsub <2 x double>
; CHECK: shufflevector <2 x double> 
define void @reorder_alt_rightsubTree(double* nocapture %c, double* noalias nocapture readonly %a, double* noalias nocapture readonly %b, double* noalias nocapture readonly %d) {
  %1 = load double, double* %a
  %2 = load double, double* %b
  %3 = fadd double %1, %2
  %4 = load double, double* %d
  %5 = fsub double %3, %4
  store double %5, double* %c
  %6 = getelementptr inbounds double, double* %d, i64 1
  %7 = load double, double* %6
  %8 = getelementptr inbounds double, double* %a, i64 1
  %9 = load double, double* %8
  %10 = getelementptr inbounds double, double* %b, i64 1
  %11 = load double, double* %10
  %12 = fadd double %9, %11
  %13 = fadd double %7, %12
  %14 = getelementptr inbounds double, double* %c, i64 1
  store double %13, double* %14
  ret void
}

; Dont vectorization of following code for float data type as sub is not commutative-
;  fc[0] = fb[0]+fa[0];
;  fc[1] = fa[1]-fb[1];
;  fc[2] = fa[2]+fb[2];
;  fc[3] = fb[3]-fa[3];
;  In the above code we can swap the 1st and 2nd operation as fadd is commutative
;  but not 2nd or 4th as fsub is not commutative. 

; CHECK-LABEL: @no_vec_shuff_reorder
; CHECK-NOT: fadd <4 x float>
; CHECK-NOT: fsub <4 x float>
; CHECK-NOT: shufflevector
define void @no_vec_shuff_reorder() #0 {
  %1 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 0), align 4
  %2 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 0), align 4
  %3 = fadd float %1, %2
  store float %3, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 0), align 4
  %4 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 1), align 4
  %5 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 1), align 4
  %6 = fsub float %4, %5
  store float %6, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 1), align 4
  %7 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 2), align 4
  %8 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 2), align 4
  %9 = fadd float %7, %8
  store float %9, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 2), align 4
  %10 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fb, i32 0, i64 3), align 4
  %11 = load float, float* getelementptr inbounds ([4 x float], [4 x float]* @fa, i32 0, i64 3), align 4
  %12 = fsub float %10, %11
  store float %12, float* getelementptr inbounds ([4 x float], [4 x float]* @fc, i32 0, i64 3), align 4
  ret void
}


attributes #0 = { nounwind }

