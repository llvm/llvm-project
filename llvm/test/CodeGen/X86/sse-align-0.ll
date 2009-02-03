; RUN: llvm-as < %s | llc -march=x86-64 | not grep mov

define <4 x float> @foo(<4 x float>* %p, <4 x float> %x) nounwind {
  %t = load <4 x float>* %p
  %z = mul <4 x float> %t, %x
  ret <4 x float> %z
}
define <2 x double> @bar(<2 x double>* %p, <2 x double> %x) nounwind {
  %t = load <2 x double>* %p
  %z = mul <2 x double> %t, %x
  ret <2 x double> %z
}
