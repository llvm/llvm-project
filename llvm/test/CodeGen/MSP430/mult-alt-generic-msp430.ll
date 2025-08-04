; RUN: llc < %s -mtriple=msp430 -no-integrated-as
; ModuleID = 'mult-alt-generic.c'
target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430"

@mout0 = common global i16 0, align 2
@min1 = common global i16 0, align 2
@marray = common global [2 x i16] zeroinitializer, align 2

define void @single_m() nounwind {
entry:
  call void asm "foo $1,$0", "=*m,*m"(ptr elementtype(i16) @mout0, ptr elementtype(i16) @min1) nounwind
  ret void
}

define void @single_o() nounwind {
entry:
  %out0 = alloca i16, align 2
  %index = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %index, align 2
  ret void
}

define void @single_V() nounwind {
entry:
  ret void
}

define void @single_lt() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r,<r"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  %tmp1 = load i16, ptr %in1, align 2
  %1 = call i16 asm "foo $1,$0", "=r,r<"(i16 %tmp1) nounwind
  store i16 %1, ptr %out0, align 2
  ret void
}

define void @single_gt() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r,>r"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  %tmp1 = load i16, ptr %in1, align 2
  %1 = call i16 asm "foo $1,$0", "=r,r>"(i16 %tmp1) nounwind
  store i16 %1, ptr %out0, align 2
  ret void
}

define void @single_r() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r,r"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  ret void
}

define void @single_i() nounwind {
entry:
  %out0 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  %0 = call i16 asm "foo $1,$0", "=r,i"(i16 1) nounwind
  store i16 %0, ptr %out0, align 2
  ret void
}

define void @single_n() nounwind {
entry:
  %out0 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  %0 = call i16 asm "foo $1,$0", "=r,n"(i16 1) nounwind
  store i16 %0, ptr %out0, align 2
  ret void
}

define void @single_E() nounwind {
entry:
  %out0 = alloca double, align 8
  store double 0.000000e+000, ptr %out0, align 8
; No lowering support.
;  %0 = call double asm "foo $1,$0", "=r,E"(double 1.000000e+001) nounwind
;  store double %0, double* %out0, align 8
  ret void
}

define void @single_F() nounwind {
entry:
  %out0 = alloca double, align 8
  store double 0.000000e+000, ptr %out0, align 8
; No lowering support.
;  %0 = call double asm "foo $1,$0", "=r,F"(double 1.000000e+000) nounwind
;  store double %0, double* %out0, align 8
  ret void
}

define void @single_s() nounwind {
entry:
  %out0 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  ret void
}

define void @single_g() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r,imr"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  %tmp1 = load i16, ptr @min1, align 2
  %1 = call i16 asm "foo $1,$0", "=r,imr"(i16 %tmp1) nounwind
  store i16 %1, ptr %out0, align 2
  %2 = call i16 asm "foo $1,$0", "=r,imr"(i16 1) nounwind
  store i16 %2, ptr %out0, align 2
  ret void
}

define void @single_X() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r,X"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  %tmp1 = load i16, ptr @min1, align 2
  %1 = call i16 asm "foo $1,$0", "=r,X"(i16 %tmp1) nounwind
  store i16 %1, ptr %out0, align 2
  %2 = call i16 asm "foo $1,$0", "=r,X"(i16 1) nounwind
  store i16 %2, ptr %out0, align 2
  %3 = call i16 asm "foo $1,$0", "=r,X"(ptr @marray) nounwind
  store i16 %3, ptr %out0, align 2
; No lowering support.
;  %4 = call i16 asm "foo $1,$0", "=r,X"(double 1.000000e+001) nounwind
;  store i16 %4, i16* %out0, align 2
;  %5 = call i16 asm "foo $1,$0", "=r,X"(double 1.000000e+000) nounwind
;  store i16 %5, i16* %out0, align 2
  ret void
}

define void @single_p() nounwind {
entry:
  %out0 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  %0 = call i16 asm "foo $1,$0", "=r,r"(ptr @marray) nounwind
  store i16 %0, ptr %out0, align 2
  ret void
}

define void @multi_m() nounwind {
entry:
  %tmp = load i16, ptr @min1, align 2
  call void asm "foo $1,$0", "=*m|r,m|r"(ptr elementtype(i16) @mout0, i16 %tmp) nounwind
  ret void
}

define void @multi_o() nounwind {
entry:
  %out0 = alloca i16, align 2
  %index = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %index, align 2
  ret void
}

define void @multi_V() nounwind {
entry:
  ret void
}

define void @multi_lt() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r|r,r|<r"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  %tmp1 = load i16, ptr %in1, align 2
  %1 = call i16 asm "foo $1,$0", "=r|r,r|r<"(i16 %tmp1) nounwind
  store i16 %1, ptr %out0, align 2
  ret void
}

define void @multi_gt() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r|r,r|>r"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  %tmp1 = load i16, ptr %in1, align 2
  %1 = call i16 asm "foo $1,$0", "=r|r,r|r>"(i16 %tmp1) nounwind
  store i16 %1, ptr %out0, align 2
  ret void
}

define void @multi_r() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r|r,r|m"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  ret void
}

define void @multi_i() nounwind {
entry:
  %out0 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  %0 = call i16 asm "foo $1,$0", "=r|r,r|i"(i16 1) nounwind
  store i16 %0, ptr %out0, align 2
  ret void
}

define void @multi_n() nounwind {
entry:
  %out0 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  %0 = call i16 asm "foo $1,$0", "=r|r,r|n"(i16 1) nounwind
  store i16 %0, ptr %out0, align 2
  ret void
}

define void @multi_E() nounwind {
entry:
  %out0 = alloca double, align 8
  store double 0.000000e+000, ptr %out0, align 8
; No lowering support.
;  %0 = call double asm "foo $1,$0", "=r|r,r|E"(double 1.000000e+001) nounwind
;  store double %0, double* %out0, align 8
  ret void
}

define void @multi_F() nounwind {
entry:
  %out0 = alloca double, align 8
  store double 0.000000e+000, ptr %out0, align 8
; No lowering support.
;  %0 = call double asm "foo $1,$0", "=r|r,r|F"(double 1.000000e+000) nounwind
;  store double %0, double* %out0, align 8
  ret void
}

define void @multi_s() nounwind {
entry:
  %out0 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  ret void
}

define void @multi_g() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r|r,r|imr"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  %tmp1 = load i16, ptr @min1, align 2
  %1 = call i16 asm "foo $1,$0", "=r|r,r|imr"(i16 %tmp1) nounwind
  store i16 %1, ptr %out0, align 2
  %2 = call i16 asm "foo $1,$0", "=r|r,r|imr"(i16 1) nounwind
  store i16 %2, ptr %out0, align 2
  ret void
}

define void @multi_X() nounwind {
entry:
  %out0 = alloca i16, align 2
  %in1 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  store i16 1, ptr %in1, align 2
  %tmp = load i16, ptr %in1, align 2
  %0 = call i16 asm "foo $1,$0", "=r|r,r|X"(i16 %tmp) nounwind
  store i16 %0, ptr %out0, align 2
  %tmp1 = load i16, ptr @min1, align 2
  %1 = call i16 asm "foo $1,$0", "=r|r,r|X"(i16 %tmp1) nounwind
  store i16 %1, ptr %out0, align 2
  %2 = call i16 asm "foo $1,$0", "=r|r,r|X"(i16 1) nounwind
  store i16 %2, ptr %out0, align 2
  %3 = call i16 asm "foo $1,$0", "=r|r,r|X"(ptr @marray) nounwind
  store i16 %3, ptr %out0, align 2
; No lowering support.
;  %4 = call i16 asm "foo $1,$0", "=r|r,r|X"(double 1.000000e+001) nounwind
;  store i16 %4, i16* %out0, align 2
;  %5 = call i16 asm "foo $1,$0", "=r|r,r|X"(double 1.000000e+000) nounwind
;  store i16 %5, i16* %out0, align 2
  ret void
}

define void @multi_p() nounwind {
entry:
  %out0 = alloca i16, align 2
  store i16 0, ptr %out0, align 2
  %0 = call i16 asm "foo $1,$0", "=r|r,r|r"(ptr @marray) nounwind
  store i16 %0, ptr %out0, align 2
  ret void
}
