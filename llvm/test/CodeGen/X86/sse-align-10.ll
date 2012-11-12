; RUN: llc < %s -march=x86-64 | grep movups | count 1

define <2 x i64> @bar(<2 x i64>* %p) nounwind {
  %t = load <2 x i64>* %p, align 8
  ret <2 x i64> %t
}
