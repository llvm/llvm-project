; RUN: llc < %s -march=x86-64 | grep movup | count 2

define void @foo(<4 x float>* %p, <4 x float> %x) nounwind {
  store <4 x float> %x, <4 x float>* %p, align 4
  ret void
}
define void @bar(<2 x double>* %p, <2 x double> %x) nounwind {
  store <2 x double> %x, <2 x double>* %p, align 8
  ret void
}
