; RUN: opt -S -deadargelim %s | FileCheck %s

define internal { i64, i64 } @f(i64 %a, i64 %b) {
start:
  %0 = insertvalue { i64, i64 } undef, i64 %a, 0
  %1 = insertvalue { i64, i64 } %0, i64 %b, 1
  ret { i64, i64 } %1
}

; Check that we don't delete either of g's return values

; CHECK-LABEL: define internal { i64, i64 } @g(i64 %a, i64 %b)
define internal { i64, i64 } @g(i64 %a, i64 %b) {
start:
  %0 = call { i64, i64 } @f(i64 %a, i64 %b)
  ret { i64, i64 } %0
}

declare dso_local i32 @test(i64, i64)

define i32 @main(i32 %argc, i8** %argv) {
start:
  %x = call { i64, i64 } @g(i64 13, i64 42)
  %x.0 = extractvalue { i64, i64 } %x, 0
  %x.1 = extractvalue { i64, i64 } %x, 1
  %z = bitcast i64 %x.0 to i64
  %y = call { i64, i64 } @f(i64 %x.0, i64 %x.1)
  %y.1 = extractvalue { i64, i64 } %y, 1
  %0 = call i32 @test(i64 %x.0, i64 %y.1)
  ret i32 %0
}

