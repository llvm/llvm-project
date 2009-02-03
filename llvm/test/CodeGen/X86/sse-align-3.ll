; RUN: llvm-as < %s | llc -march=x86-64 | grep movap | count 2

define void @foo(<4 x float>* %p, <4 x float> %x) nounwind {
  store <4 x float> %x, <4 x float>* %p
  ret void
}
define void @bar(<2 x double>* %p, <2 x double> %x) nounwind {
  store <2 x double> %x, <2 x double>* %p
  ret void
}
