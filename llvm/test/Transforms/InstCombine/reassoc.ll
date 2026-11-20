; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; CHECK-LABEL: define i8 @newDenseMat(
; CHECK: %3 = shl i8 %1, 1
; CHECK: %4 = mul i8 %0, %3

define i8 @newDenseMat(i8 %0, i8 %1, i8* %2) {
entry:
  %4 = shl i8 %1, 1
  %5 = shl i8 %0, 1
  %6 = mul i8 %5, %1
  store i8 %6, i8* %2, align 8
  ret i8 %4
}

; CHECK-LABEL: define i32 @H5HF__sect_indirect_init_rows(
; CHECK: %3 = add i32 %2, 1
; CHECK: %4 = sub i32 %3, %1

define noundef i32 @H5HF__sect_indirect_init_rows(i32* %0, i32 %1, i32 %2) {
entry:
  %4 = add i32 %2, 1
  store i32 %4, i32* %0, align 8
  %reass.sub = sub i32 %2, %1
  %5 = add i32 %reass.sub, 1
  ret i32 %5
}

define noundef i32 @tgt2(i32* %0, i32 %1, i32 %2) {
entry:
  %4 = add i32 %2, 1
  store i32 %4, i32* %0, align 8
  %5 = sub i32 %4, %1
  ret i32 %5
}
