; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ALL
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=ALL

; FIXME Add tests for thumbv7, they currently fail MI verification because
;       of a mismatch in register classes in uses.

; This test verifies that load/store instructions are properly generated,
; and that they pass MI verification (wasn't the case until 2013-06-08).

@a = global i8 1, align 1
@b = global i16 2, align 2
@c = global i32 4, align 4

; ldr

define i8 @t1() nounwind uwtable ssp {
; ALL: @t1
; ALL: ldrb
; ALL: add
  %1 = load i8, ptr @a, align 1
  %2 = add nsw i8 %1, 1
  ret i8 %2
}

define i16 @t2() nounwind uwtable ssp {
; ALL: @t2
; ALL: ldrh
; ALL: add
  %1 = load i16, ptr @b, align 2
  %2 = add nsw i16 %1, 1
  ret i16 %2
}

define i32 @t3() nounwind uwtable ssp {
; ALL: @t3
; ALL: ldr
; ALL: add
  %1 = load i32, ptr @c, align 4
  %2 = add nsw i32 %1, 1
  ret i32 %2
}

; str

define void @t4(i8 %v) nounwind uwtable ssp {
; ALL: @t4
; ALL: add
; ALL: strb
  %1 = add nsw i8 %v, 1
  store i8 %1, ptr @a, align 1
  ret void
}

define void @t5(i16 %v) nounwind uwtable ssp {
; ALL: @t5
; ALL: add
; ALL: strh
  %1 = add nsw i16 %v, 1
  store i16 %1, ptr @b, align 2
  ret void
}

define void @t6(i32 %v) nounwind uwtable ssp {
; ALL: @t6
; ALL: add
; ALL: str
  %1 = add nsw i32 %v, 1
  store i32 %1, ptr @c, align 4
  ret void
}
