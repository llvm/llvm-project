; RUN: llc -mtriple=x86_64 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

define i64 @test1(i64 %i) nounwind readnone {
  %loc = alloca i64
  %j = load i64, ptr %loc
  %r = add i64 %i, %j
  ret i64 %r
}

define i64 @test2(i32 %i) nounwind readnone {
  %loc = alloca i32
  %j = load i32, ptr %loc
  %r = add i32 %i, %j
  %ext = zext i32 %r to i64
  ret i64 %ext
}
