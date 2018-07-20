; RUN: llc -march=hexagon -O3 -hexagon-small-data-threshold=0 < %s | FileCheck %s
; This lit test validates that storetrunc for a 64bit value picks a store
; absolute pattern instead of base + index store pattern. This will facilitate
; the constant extender optimization pass to move the immediate value to a register
; if there are more than two uses and replace all the uses of the constant.
; Generation of absolute pattern for a 64 bit truncated value also aviods an
; extra move.

@g0 = external global i8, align 8
@g1 = external global i16, align 8
@g2 = external global i32, align 8

; CHECK-LABEL: f0:
; CHECK: memd(##441656) = r{{[0-9]+}}
define void @f0(i64 %a0) #0 {
b0:
  store volatile i64 %a0, i64* inttoptr (i32 441656 to i64*)
  ret void
}

; CHECK-LABEL: f1:
; CHECK: memw(##441656) = r{{[0-9]+}}
define void @f1(i64 %a0) #0 {
b0:
  %v0 = trunc i64 %a0 to i32
  store volatile i32 %v0, i32* inttoptr (i32 441656 to i32*)
  ret void
}

; CHECK-LABEL: f2:
; CHECK: memh(##441656) = r{{[0-9]+}}
define void @f2(i64 %a0) #0 {
b0:
  %v0 = trunc i64 %a0 to i16
  store volatile i16 %v0, i16* inttoptr (i32 441656 to i16*)
  ret void
}

; CHECK-LABEL: f3:
; CHECK: memb(##441656) = r{{[0-9]+}}
define void @f3(i64 %a0) #0 {
b0:
  %v0 = trunc i64 %a0 to i8
  store volatile i8 %v0, i8* inttoptr (i32 441656 to i8*)
  ret void
}

; CHECK-LABEL: f4:
; CHECK: memw(##g2) = r{{[0-9]+}}
define void @f4(i64 %a0) #0 {
b0:
  %v0 = trunc i64 %a0 to i32
  store volatile i32 %v0, i32* @g2
  ret void
}

; CHECK-LABEL: f5:
; CHECK: memh(##g1) = r{{[0-9]+}}
define void @f5(i64 %a0) #0 {
b0:
  %v0 = trunc i64 %a0 to i16
  store volatile i16 %v0, i16* @g1
  ret void
}

; CHECK-LABEL: f6:
; CHECK: memb(##g0) = r{{[0-9]+}}
define void @f6(i64 %a0) #0 {
b0:
  %v0 = trunc i64 %a0 to i8
  store volatile i8 %v0, i8* @g0
  ret void
}

attributes #0 = { nounwind }
