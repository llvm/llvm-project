; RUN: llvm-as < %s | llc -march=x86-64 | grep movaps | count 1

define void @bar(<2 x i64>* %p, <2 x i64> %x) nounwind {
  store <2 x i64> %x, <2 x i64>* %p
  ret void
}
