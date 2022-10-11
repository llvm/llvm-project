; RUN: llc -relocation-model=static < %s -O0 -verify-machineinstrs -fast-isel -fast-isel-abort=1 -mattr=-vsx -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64
; RUN: llc -relocation-model=static < %s -O0 -verify-machineinstrs -fast-isel -fast-isel-abort=1 -mattr=+vsx -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=VSX
; RUN: llc -relocation-model=static < %s -O0 -verify-machineinstrs -fast-isel -fast-isel-abort=1 -mattr=spe  -mtriple=powerpc-unknown-linux-gnu -mcpu=e500 | FileCheck %s --check-prefix=SPE

; This test verifies that load/store instructions are properly generated,
; and that they pass MI verification.

@a = dso_local global i8 1, align 1
@b = dso_local global i16 2, align 2
@c = dso_local global i32 4, align 4
@d = dso_local global i64 8, align 8
@e = dso_local global float 1.25, align 4
@f = dso_local global double 3.5, align 8

%struct.s = type<{ i8, i32 }>
%struct.t = type<{ i8, i64 }>

@g = dso_local global %struct.s <{ i8 1, i32 2 }>, align 1
@h = dso_local global %struct.t <{ i8 1, i64 2 }>, align 1

@i = common dso_local global [8192 x i64] zeroinitializer, align 8

; load

define i8 @t1() nounwind {
; ELF64-LABEL: t1:
  %1 = load i8, ptr @a, align 1
; ELF64: lbz
  %2 = add nsw i8 %1, 1
; ELF64: addi
  ret i8 %2
}

define i16 @t2() nounwind {
; ELF64-LABEL: t2:
  %1 = load i16, ptr @b, align 2
; ELF64: lhz
  %2 = add nsw i16 %1, 1
; ELF64: addi
  ret i16 %2
}

define dso_local i32 @t3() nounwind {
; ELF64-LABEL: t3:
  %1 = load i32, ptr @c, align 4
; ELF64: lwz
  %2 = add nsw i32 %1, 1
; ELF64: addi
  ret i32 %2
}

define i64 @t4() nounwind {
; ELF64-LABEL: t4:
  %1 = load i64, ptr @d, align 4
; ELF64: ld
  %2 = add nsw i64 %1, 1
; ELF64: addi
  ret i64 %2
}

define dso_local float @t5() nounwind {
; ELF64-LABEL: t5:
; SPE-LABEL: t5:
  %1 = load float, ptr @e, align 4
; ELF64: lfs
; SPE: lwz
  %2 = fadd float %1, 1.0
; ELF64: fadds
; SPE: efsadd
  ret float %2
}

define dso_local double @t6() nounwind {
; ELF64-LABEL: t6:
; SPE-LABEL: t6:
  %1 = load double, ptr @f, align 8
; ELF64: lfd
; VSX: lxsdx
; SPE: evldd
  %2 = fadd double %1, 1.0
; ELF64: fadd
; VSX: xsadddp
; SPE: efdadd
  ret double %2
}

; store

define dso_local void @t7(i8 %v) nounwind {
; ELF64-LABEL: t7:
  %1 = add nsw i8 %v, 1
  store i8 %1, ptr @a, align 1
; ELF64: addi
; ELF64: addis
; ELF64: addi
; ELF64: stb
  ret void
}

define dso_local void @t8(i16 %v) nounwind {
; ELF64-LABEL: t8:
  %1 = add nsw i16 %v, 1
  store i16 %1, ptr @b, align 2
; ELF64: addi
; ELF64: addis
; ELF64: addi
; ELF64: sth
  ret void
}

define dso_local void @t9(i32 %v) nounwind {
; ELF64-LABEL: t9:
  %1 = add nsw i32 %v, 1
  store i32 %1, ptr @c, align 4
; ELF64: addi
; ELF64: addis
; ELF64: addi
; ELF64: stw
  ret void
}

define dso_local void @t10(i64 %v) nounwind {
; ELF64-LABEL: t10:
  %1 = add nsw i64 %v, 1
  store i64 %1, ptr @d, align 4
; ELF64: addi
; ELF64: addis
; ELF64: addi
; ELF64: std
  ret void
}

define dso_local void @t11(float %v) nounwind {
; ELF64-LABEL: t11:
; SPE-LABEL: t11:
  %1 = fadd float %v, 1.0
  store float %1, ptr @e, align 4
; ELF64: fadds
; ELF64: stfs
; SPE: efsadd
; SPE: stw
  ret void
}

define dso_local void @t12(double %v) nounwind {
; ELF64-LABEL: t12:
; SPE-LABEL: t12:
  %1 = fadd double %v, 1.0
  store double %1, ptr @f, align 8
; ELF64: fadd
; ELF64: stfd
; VSX: xsadddp
; VSX: stxsdx
; SPE: efdadd
; SPE: evstdd
  ret void
}

;; lwa requires an offset divisible by 4, so we need lwax here.
define i64 @t13() nounwind {
; ELF64-LABEL: t13:
  %1 = load i32, ptr getelementptr inbounds (%struct.s, ptr @g, i32 0, i32 1), align 1
  %2 = sext i32 %1 to i64
; ELF64: li
; ELF64: lwax
  %3 = add nsw i64 %2, 1
; ELF64: addi
  ret i64 %3
}

;; ld requires an offset divisible by 4, so we need ldx here.
define i64 @t14() nounwind {
; ELF64-LABEL: t14:
  %1 = load i64, ptr getelementptr inbounds (%struct.t, ptr @h, i32 0, i32 1), align 1
; ELF64: li
; ELF64: ldx
  %2 = add nsw i64 %1, 1
; ELF64: addi
  ret i64 %2
}

;; std requires an offset divisible by 4, so we need stdx here.
define dso_local void @t15(i64 %v) nounwind {
; ELF64-LABEL: t15:
  %1 = add nsw i64 %v, 1
  store i64 %1, ptr getelementptr inbounds (%struct.t, ptr @h, i32 0, i32 1), align 1
; ELF64: addi
; ELF64: addis
; ELF64: addi
; ELF64: li
; ELF64: stdx
  ret void
}

;; ld requires an offset that fits in 16 bits, so we need ldx here.
define i64 @t16() nounwind {
; ELF64-LABEL: t16:
  %1 = load i64, ptr getelementptr inbounds ([8192 x i64], ptr @i, i32 0, i64 5000), align 8
; ELF64: lis
; ELF64: ori
; ELF64: ldx
  %2 = add nsw i64 %1, 1
; ELF64: addi
  ret i64 %2
}

;; std requires an offset that fits in 16 bits, so we need stdx here.
define dso_local void @t17(i64 %v) nounwind {
; ELF64-LABEL: t17:
  %1 = add nsw i64 %v, 1
  store i64 %1, ptr getelementptr inbounds ([8192 x i64], ptr @i, i32 0, i64 5000), align 8
; ELF64: addi
; ELF64: addis
; ELF64: addi
; ELF64: lis
; ELF64: ori
; ELF64: stdx
  ret void
}

