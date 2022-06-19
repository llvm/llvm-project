; RUN: llc -mtriple=x86_64 -stop-after=finalize-isel -debug-only=isel -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=PIC
; RUN: llc -mtriple=x86_64-windows -stop-after=finalize-isel -debug-only=isel -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=WIN

define i64 @i64_test(i64 %i) nounwind readnone {
  %loc = alloca i64
  %j = load i64, i64 * %loc
  %r = add i64 %i, %j
  ret i64 %r
}

define i64 @i32_test(i32 %i) nounwind readnone {
  %loc = alloca i32
  %j = load i32, i32 * %loc
  %r = add i32 %i, %j
  %ext = zext i32 %r to i64
  ret i64 %ext
}

define i64 @i16_test(i16 %i) nounwind readnone {
  %loc = alloca i16
  %j = load i16, i16 * %loc
  %r = add i16 %i, %j
  %ext = zext i16 %r to i64
  ret i64 %ext
}

define i64 @i8_test(i8 %i) nounwind readnone {
  %loc = alloca i8
  %j = load i8, i8 * %loc
  %r = add i8 %i, %j
  %ext = zext i8 %r to i64
  ret i64 %ext
}
