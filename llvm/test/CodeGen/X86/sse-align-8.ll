; RUN: llc < %s -mtriple=x86_64-- | grep movups | count 1

define void @bar(<2 x i64>* %p, <2 x i64> %x) nounwind {
  store <2 x i64> %x, <2 x i64>* %p, align 8
  ret void
}
