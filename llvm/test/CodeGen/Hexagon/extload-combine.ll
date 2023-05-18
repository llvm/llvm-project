; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 -disable-hsdr  < %s | FileCheck %s
; Check that the combine/stxw instructions are being generated.
; In case of combine one of the operand should be 0 and another should be
; the output of absolute addressing load instruction.

@a = external global i16
@b = external global i16
@c = external global i16
@char_a = external global i8
@char_b = external global i8
@char_c = external global i8
@int_a = external global i32
@int_b = external global i32
@int_c = external global i32

; Function Attrs: nounwind
define i64 @short_test1() #0 {
; CHECK: [[VAR:r[0-9]+]] = memuh(##
; CHECK: combine(#0,[[VAR]])
entry:
  store i16 0, ptr @a, align 2
  %0 = load i16, ptr @b, align 2
  %conv2 = zext i16 %0 to i64
  ret i64 %conv2
}

; Function Attrs: nounwind
define i64 @short_test2() #0 {
; CHECK: [[VAR1:r[0-9]+]] = memh(##
; CHECK: sxtw([[VAR1]])
entry:
  store i16 0, ptr @a, align 2
  %0 = load i16, ptr @c, align 2
  %conv2 = sext i16 %0 to i64
  ret i64 %conv2
}

; Function Attrs: nounwind
define i64 @char_test1() #0 {
; CHECK: [[VAR2:r[0-9]+]] = memub(##
; CHECK: combine(#0,[[VAR2]])
entry:
  store i8 0, ptr @char_a, align 1
  %0 = load i8, ptr @char_b, align 1
  %conv2 = zext i8 %0 to i64
  ret i64 %conv2
}

; Function Attrs: nounwind
define i64 @char_test2() #0 {
; CHECK: [[VAR3:r[0-9]+]] = memb(##
; CHECK: sxtw([[VAR3]])
entry:
  store i8 0, ptr @char_a, align 1
  %0 = load i8, ptr @char_c, align 1
  %conv2 = sext i8 %0 to i64
  ret i64 %conv2
}

; Function Attrs: nounwind
define i64 @int_test1() #0 {
; CHECK: [[VAR4:r[0-9]+]] = memw(##
; CHECK: combine(#0,[[VAR4]])
entry:
  store i32 0, ptr @int_a, align 4
  %0 = load i32, ptr @int_b, align 4
  %conv = zext i32 %0 to i64
  ret i64 %conv
}

; Function Attrs: nounwind
define i64 @int_test2() #0 {
; CHECK: [[VAR5:r[0-9]+]] = memw(##
; CHECK: sxtw([[VAR5]])
entry:
  store i32 0, ptr @int_a, align 4
  %0 = load i32, ptr @int_c, align 4
  %conv = sext i32 %0 to i64
  ret i64 %conv
}
