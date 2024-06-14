; RUN: llc -o - %s -verify-machineinstrs
; This used to crash when coalescing a regclass like GR16 which did not support
; the sub_8bit_hi subregister with a class like GR16_ABCD that did.
target triple = "x86_64-apple-macosx10.10.0"

define void @test() #0 {
entry:
  br i1 undef, label %loop, label %for.end597

loop:
  %0 = load i16, ptr null, align 4
  %1 = load i16, ptr undef, align 4
  %or1 = or i16 %1, %0
  %or2 = trunc i16 %or1 to i8
  store i8 %or2, ptr undef, align 4
  %2 = or i16 %1, %0
  %or3 = lshr i16 %2, 8
  %or4 = trunc i16 %or3 to i8
  store i8 %or4, ptr undef, align 1
  %3 = load i32, ptr undef, align 4
  %4 = load i32, ptr undef, align 4
  %or5 = or i32 %4, %3
  store i32 %or5, ptr undef, align 4
  store i32 0, ptr undef, align 4
  br label %loop

for.end597:
  ret void
}
