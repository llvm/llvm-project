; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-darwin | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=THUMB

@a = global i8 1, align 1
@b = global i16 2, align 2

define void @t1() nounwind uwtable ssp {
; ARM: t1
; ARM: ldrb
; ARM-NOT: uxtb
; THUMB: t1
; THUMB: ldrb
; THUMB-NOT: uxtb
  %1 = load i8* @a, align 1
  call void @foo1(i8 zeroext %1)
  ret void
}

define void @t2() nounwind uwtable ssp {
; ARM: t2
; ARM: ldrh
; ARM-NOT: uxth
; THUMB: t2
; THUMB: ldrh
; THUMB-NOT: uxth
  %1 = load i16* @b, align 2
  call void @foo2(i16 zeroext %1)
  ret void
}

declare void @foo1(i8 zeroext)
declare void @foo2(i16 zeroext)

define i32 @t3() nounwind uwtable ssp {
; ARM: t3
; ARM: ldrb
; ARM-NOT: uxtb
; THUMB: t3
; THUMB: ldrb
; THUMB-NOT: uxtb
  %1 = load i8* @a, align 1
  %2 = zext i8 %1 to i32
  ret i32 %2
}

define i32 @t4() nounwind uwtable ssp {
; ARM: t4
; ARM: ldrh
; ARM-NOT: uxth
; THUMB: t4
; THUMB: ldrh
; THUMB-NOT: uxth
  %1 = load i16* @b, align 2
  %2 = zext i16 %1 to i32
  ret i32 %2
}

define i32 @t5() nounwind uwtable ssp {
; ARM: t5
; ARM: ldrsh
; ARM-NOT: sxth
; THUMB: t5
; THUMB: ldrsh
; THUMB-NOT: sxth
  %1 = load i16* @b, align 2
  %2 = sext i16 %1 to i32
  ret i32 %2
}

define i32 @t6() nounwind uwtable ssp {
; ARM: t6
; ARM: ldrsb
; ARM-NOT: sxtb
; THUMB: t6
; THUMB: ldrsb
; THUMB-NOT: sxtb
  %1 = load i8* @a, align 2
  %2 = sext i8 %1 to i32
  ret i32 %2
}
