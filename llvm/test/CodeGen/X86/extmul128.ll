; RUN: llvm-as < %s | llc -march=x86-64 | grep mul | count 2

define i128 @i64_sext_i128(i64 %a, i64 %b) {
  %aa = sext i64 %a to i128
  %bb = sext i64 %b to i128
  %cc = mul i128 %aa, %bb
  ret i128 %cc
}
define i128 @i64_zext_i128(i64 %a, i64 %b) {
  %aa = zext i64 %a to i128
  %bb = zext i64 %b to i128
  %cc = mul i128 %aa, %bb
  ret i128 %cc
}
