; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64

@a = global i8 1, align 1
@b = global i16 2, align 2
@c = global i32 4, align 4

define void @t1() nounwind {
; PPC64: t1
  %1 = load i8, i8* @a, align 1
  call void @foo1(i8 zeroext %1)
; PPC64: lbz
; PPC64-NOT: rldicl
; PPC64-NOT: rlwinm
  ret void
}

define void @t2() nounwind {
; PPC64: t2
  %1 = load i16, i16* @b, align 2
  call void @foo2(i16 zeroext %1)
; PPC64: lhz
; PPC64-NOT: rldicl
; PPC64-NOT: rlwinm
  ret void
}

define void @t2a() nounwind {
; PPC64: t2a
  %1 = load i32, i32* @c, align 4
  call void @foo3(i32 zeroext %1)
; PPC64: lwz
; PPC64-NOT: rldicl
; PPC64-NOT: rlwinm
  ret void
}

declare void @foo1(i8 zeroext)
declare void @foo2(i16 zeroext)
declare void @foo3(i32 zeroext)

define i32 @t3() nounwind {
; PPC64: t3
  %1 = load i8, i8* @a, align 1
  %2 = zext i8 %1 to i32
; PPC64: lbz
; PPC64-NOT: rlwinm
  ret i32 %2
}

define i32 @t4() nounwind {
; PPC64: t4
  %1 = load i16, i16* @b, align 2
  %2 = zext i16 %1 to i32
; PPC64: lhz
; PPC64-NOT: rlwinm
  ret i32 %2
}

define i32 @t5() nounwind {
; PPC64: t5
  %1 = load i16, i16* @b, align 2
  %2 = sext i16 %1 to i32
; PPC64: lha
; PPC64-NOT: rlwinm
  ret i32 %2
}

define i32 @t6() nounwind {
; PPC64: t6
  %1 = load i8, i8* @a, align 2
  %2 = sext i8 %1 to i32
; PPC64: lbz
; PPC64-NOT: rlwinm
  ret i32 %2
}

define i64 @t7() nounwind {
; PPC64: t7
  %1 = load i8, i8* @a, align 1
  %2 = zext i8 %1 to i64
; PPC64: lbz
; PPC64-NOT: rldicl
  ret i64 %2
}

define i64 @t8() nounwind {
; PPC64: t8
  %1 = load i16, i16* @b, align 2
  %2 = zext i16 %1 to i64
; PPC64: lhz
; PPC64-NOT: rldicl
  ret i64 %2
}

define i64 @t9() nounwind {
; PPC64: t9
  %1 = load i16, i16* @b, align 2
  %2 = sext i16 %1 to i64
; PPC64: lha
; PPC64-NOT: extsh
  ret i64 %2
}

define i64 @t10() nounwind {
; PPC64: t10
  %1 = load i8, i8* @a, align 2
  %2 = sext i8 %1 to i64
; PPC64: lbz
; PPC64: extsb
  ret i64 %2
}

define i64 @t11() nounwind {
; PPC64: t11
  %1 = load i32, i32* @c, align 4
  %2 = zext i32 %1 to i64
; PPC64: lwz
; PPC64-NOT: rldicl
  ret i64 %2
}

define i64 @t12() nounwind {
; PPC64: t12
  %1 = load i32, i32* @c, align 4
  %2 = sext i32 %1 to i64
; PPC64: lwa
; PPC64-NOT: extsw
  ret i64 %2
}
