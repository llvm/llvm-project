; RUN: llc -march=hexagon -O3 -hexagon-small-data-threshold=0 < %s | FileCheck %s
; Check that absolute loads are generated for 64-bit

target triple = "hexagon-unknown--elf"

@g0 = external global i8, align 8
@g1 = external global i16, align 8
@g2 = external global i32, align 8
@g3 = external global i64, align 8

; CHECK-LABEL: f0:
; CHECK: = memd(##441656)
define i64 @f0() #0 {
b0:
  %v0 = load volatile i64, ptr inttoptr (i32 441656 to ptr)
  ret i64 %v0
}

; CHECK-LABEL: f1:
; CHECK: = memw(##441656)
define i64 @f1() #0 {
b0:
  %v0 = load volatile i32, ptr inttoptr (i32 441656 to ptr)
  %v1 = sext i32 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f2:
; CHECK: = memw(##441656)
define i64 @f2() #0 {
b0:
  %v0 = load volatile i32, ptr inttoptr (i32 441656 to ptr)
  %v1 = zext i32 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f3:
; CHECK: = memh(##441656)
define i64 @f3() #0 {
b0:
  %v0 = load volatile i16, ptr inttoptr (i32 441656 to ptr)
  %v1 = sext i16 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f4:
; CHECK: = memuh(##441656)
define i64 @f4() #0 {
b0:
  %v0 = load volatile i16, ptr inttoptr (i32 441656 to ptr)
  %v1 = zext i16 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f5:
; CHECK: = memb(##441656)
define i64 @f5() #0 {
b0:
  %v0 = load volatile i8, ptr inttoptr (i32 441656 to ptr)
  %v1 = sext i8 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f6:
; CHECK: = memub(##441656)
define i64 @f6() #0 {
b0:
  %v0 = load volatile i8, ptr inttoptr (i32 441656 to ptr)
  %v1 = zext i8 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f7:
; CHECK: = memd(##g3)
define i64 @f7() #0 {
b0:
  %v0 = load volatile i64, ptr @g3
  ret i64 %v0
}

; CHECK-LABEL: f8:
; CHECK: = memw(##g2)
define i64 @f8() #0 {
b0:
  %v0 = load volatile i32, ptr @g2
  %v1 = sext i32 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f9:
; CHECK: = memw(##g2)
define i64 @f9() #0 {
b0:
  %v0 = load volatile i32, ptr @g2
  %v1 = zext i32 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f10:
; CHECK: = memh(##g1)
define i64 @f10() #0 {
b0:
  %v0 = load volatile i16, ptr @g1
  %v1 = sext i16 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f11:
; CHECK: = memuh(##g1)
define i64 @f11() #0 {
b0:
  %v0 = load volatile i16, ptr @g1
  %v1 = zext i16 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f12:
; CHECK: = memb(##g0)
define i64 @f12() #0 {
b0:
  %v0 = load volatile i8, ptr @g0
  %v1 = sext i8 %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: f13:
; CHECK: = memub(##g0)
define i64 @f13() #0 {
b0:
  %v0 = load volatile i8, ptr @g0
  %v1 = zext i8 %v0 to i64
  ret i64 %v1
}

attributes #0 = { nounwind }
