; RUN: llvm-as < %s | llc -march=x86 | grep xor | count 2

define i64 @foo() nounwind {
  ret i64 0
}
