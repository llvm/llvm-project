; RUN: llc < %s -march=x86-64 | grep mul | count 3

define i128 @foo(i128 %t, i128 %u) {
  %k = mul i128 %t, %u
  ret i128 %k
}
