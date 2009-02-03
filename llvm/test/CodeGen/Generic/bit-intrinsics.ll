; Make sure this testcase is supported by all code generators. Either the
; intrinsic is supported natively or IntrinsicLowering provides it.
; RUN: llvm-as < %s > %t.bc
; RUN: lli --force-interpreter=true %t.bc

declare i32 @llvm.part.set.i32.i32(i32 %x, i32 %rep, i32 %hi, i32 %lo)
declare i16 @llvm.part.set.i16.i16(i16 %x, i16 %rep, i32 %hi, i32 %lo)
define i32 @test_part_set(i32 %A, i16 %B) {
  %a = call i32 @llvm.part.set.i32.i32(i32 %A, i32 27, i32 8, i32 0)
  %b = call i16 @llvm.part.set.i16.i16(i16 %B, i16 27, i32 8, i32 0)
  %c = zext i16 %b to i32
  %d = add i32 %a, %c
  ret i32 %d
}

declare i32 @llvm.part.select.i32(i32 %x, i32 %hi, i32 %lo)
declare i16 @llvm.part.select.i16(i16 %x, i32 %hi, i32 %lo)
define i32 @test_part_select(i32 %A, i16 %B) {
  %a = call i32 @llvm.part.select.i32(i32 %A, i32 8, i32 0)
  %b = call i16 @llvm.part.select.i16(i16 %B, i32 8, i32 0)
  %c = zext i16 %b to i32
  %d = add i32 %a, %c
  ret i32 %d
}

define i32 @main(i32 %argc, i8** %argv) {
  %a = call i32 @test_part_set(i32 23, i16 57)
  %b = call i32 @test_part_select(i32 23, i16 57)
  %c = add i32 %a, %b
  %d = urem i32 %c, 1
  ret i32 %d
}
