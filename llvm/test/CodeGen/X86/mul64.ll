; RUN: llc < %s -march=x86 | grep mul | count 3

define i64 @foo(i64 %t, i64 %u) {
  %k = mul i64 %t, %u
  ret i64 %k
}
