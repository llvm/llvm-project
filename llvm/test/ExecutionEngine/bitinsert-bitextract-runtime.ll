; RUN: %lli %s | FileCheck %s
; CHECK: extract=0x7812 insert=0xef3456be

@.str = private unnamed_addr constant [30 x i8] c"extract=0x%04x insert=0x%08x\0A\00", align 1

declare i32 @printf(ptr, ...)

define i16 @test_bitextract_var(b32 %src, i32 %off) noinline {
  %result = bitextract i16, b32 %src, i32 %off
  ret i16 %result
}

define b32 @test_bitinsert_var(b32 %base, i16 %val, i32 %off) noinline {
  %result = bitinsert b32 %base, i16 %val, i32 %off
  ret b32 %result
}

define i32 @main() {
  ; Test values: 305419896 is 0x12345678, 48879 is 0xBEEF
  %extract = call i16 @test_bitextract_var(b32 305419896, i32 8)
  %insert = call b32 @test_bitinsert_var(b32 305419896, i16 48879, i32 8)

  %extract32 = zext i16 %extract to i32
  %insert32 = bitcast b32 %insert to i32

  %fmt = getelementptr inbounds [30 x i8], ptr @.str, i64 0, i64 0
  %call = call i32 (ptr, ...) @printf(ptr %fmt, i32 %extract32, i32 %insert32)
  ret i32 0
}
