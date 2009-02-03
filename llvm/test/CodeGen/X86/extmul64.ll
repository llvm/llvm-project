; RUN: llvm-as < %s | llc -march=x86 | grep mul | count 2

define i64 @i32_sext_i64(i32 %a, i32 %b) {
  %aa = sext i32 %a to i64
  %bb = sext i32 %b to i64
  %cc = mul i64 %aa, %bb
  ret i64 %cc
}
define i64 @i32_zext_i64(i32 %a, i32 %b) {
  %aa = zext i32 %a to i64
  %bb = zext i32 %b to i64
  %cc = mul i64 %aa, %bb
  ret i64 %cc
}
