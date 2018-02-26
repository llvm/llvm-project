; RUN: llc < %s -mtriple=i686-- | grep xor | count 2

define i64 @foo() nounwind {
  ret i64 0
}
