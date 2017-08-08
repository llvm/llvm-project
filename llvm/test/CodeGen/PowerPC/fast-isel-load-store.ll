; FIXME: FastISel currently returns false if it hits code that uses VSX
; registers and with -fast-isel-abort=1 turned on the test case will then fail.
; When fastisel better supports VSX fix up this test case.
;
; RUN: llc -relocation-model=static < %s -O0 -verify-machineinstrs -fast-isel -fast-isel-abort=1 -mattr=-vsx -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

; This test verifies that load/store instructions are properly generated,
; and that they pass MI verification.

@a = global i8 1, align 1
@b = global i16 2, align 2
@c = global i32 4, align 4
@d = global i64 8, align 8
@e = global float 1.25, align 4
@f = global double 3.5, align 8

%struct.s = type<{ i8, i32 }>
%struct.t = type<{ i8, i64 }>

@g = global %struct.s <{ i8 1, i32 2 }>, align 1
@h = global %struct.t <{ i8 1, i64 2 }>, align 1

@i = common global [8192 x i64] zeroinitializer, align 8

; load

define i8 @t1() nounwind {
; ELF64: t1
  %1 = load i8, i8* @a, align 1
; ELF64: lbz
  %2 = add nsw i8 %1, 1
; ELF64: addi
  ret i8 %2
}

define i16 @t2() nounwind {
; ELF64: t2
  %1 = load i16, i16* @b, align 2
; ELF64: lhz
  %2 = add nsw i16 %1, 1
; ELF64: addi
  ret i16 %2
}

define i32 @t3() nounwind {
; ELF64: t3
  %1 = load i32, i32* @c, align 4
; ELF64: lwz
  %2 = add nsw i32 %1, 1
; ELF64: addi
  ret i32 %2
}

define i64 @t4() nounwind {
; ELF64: t4
  %1 = load i64, i64* @d, align 4
; ELF64: ld
  %2 = add nsw i64 %1, 1
; ELF64: addi
  ret i64 %2
}

define float @t5() nounwind {
; ELF64: t5
  %1 = load float, float* @e, align 4
; ELF64: lfs
  %2 = fadd float %1, 1.0
; ELF64: fadds
  ret float %2
}

define double @t6() nounwind {
; ELF64: t6
  %1 = load double, double* @f, align 8
; ELF64: lfd
  %2 = fadd double %1, 1.0
; ELF64: fadd
  ret double %2
}

; store

define void @t7(i8 %v) nounwind {
; ELF64: t7
  %1 = add nsw i8 %v, 1
  store i8 %1, i8* @a, align 1
; ELF64: addis
; ELF64: addi
; ELF64: addi
; ELF64: stb
  ret void
}

define void @t8(i16 %v) nounwind {
; ELF64: t8
  %1 = add nsw i16 %v, 1
  store i16 %1, i16* @b, align 2
; ELF64: addis
; ELF64: addi
; ELF64: addi
; ELF64: sth
  ret void
}

define void @t9(i32 %v) nounwind {
; ELF64: t9
  %1 = add nsw i32 %v, 1
  store i32 %1, i32* @c, align 4
; ELF64: addis
; ELF64: addi
; ELF64: addi
; ELF64: stw
  ret void
}

define void @t10(i64 %v) nounwind {
; ELF64: t10
  %1 = add nsw i64 %v, 1
  store i64 %1, i64* @d, align 4
; ELF64: addis
; ELF64: addi
; ELF64: addi
; ELF64: std
  ret void
}

define void @t11(float %v) nounwind {
; ELF64: t11
  %1 = fadd float %v, 1.0
  store float %1, float* @e, align 4
; ELF64: fadds
; ELF64: stfs
  ret void
}

define void @t12(double %v) nounwind {
; ELF64: t12
  %1 = fadd double %v, 1.0
  store double %1, double* @f, align 8
; ELF64: fadd
; ELF64: stfd
  ret void
}

;; lwa requires an offset divisible by 4, so we need lwax here.
define i64 @t13() nounwind {
; ELF64: t13
  %1 = load i32, i32* getelementptr inbounds (%struct.s, %struct.s* @g, i32 0, i32 1), align 1
  %2 = sext i32 %1 to i64
; ELF64: li
; ELF64: lwax
  %3 = add nsw i64 %2, 1
; ELF64: addi
  ret i64 %3
}

;; ld requires an offset divisible by 4, so we need ldx here.
define i64 @t14() nounwind {
; ELF64: t14
  %1 = load i64, i64* getelementptr inbounds (%struct.t, %struct.t* @h, i32 0, i32 1), align 1
; ELF64: li
; ELF64: ldx
  %2 = add nsw i64 %1, 1
; ELF64: addi
  ret i64 %2
}

;; std requires an offset divisible by 4, so we need stdx here.
define void @t15(i64 %v) nounwind {
; ELF64: t15
  %1 = add nsw i64 %v, 1
  store i64 %1, i64* getelementptr inbounds (%struct.t, %struct.t* @h, i32 0, i32 1), align 1
; ELF64: addis
; ELF64: addi
; ELF64: addi
; ELF64: li
; ELF64: stdx
  ret void
}

;; ld requires an offset that fits in 16 bits, so we need ldx here.
define i64 @t16() nounwind {
; ELF64: t16
  %1 = load i64, i64* getelementptr inbounds ([8192 x i64], [8192 x i64]* @i, i32 0, i64 5000), align 8
; ELF64: lis
; ELF64: ori
; ELF64: ldx
  %2 = add nsw i64 %1, 1
; ELF64: addi
  ret i64 %2
}

;; std requires an offset that fits in 16 bits, so we need stdx here.
define void @t17(i64 %v) nounwind {
; ELF64: t17
  %1 = add nsw i64 %v, 1
  store i64 %1, i64* getelementptr inbounds ([8192 x i64], [8192 x i64]* @i, i32 0, i64 5000), align 8
; ELF64: addis
; ELF64: addi
; ELF64: addi
; ELF64: lis
; ELF64: ori
; ELF64: stdx
  ret void
}

