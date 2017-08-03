; RUN: llc < %s -march=x86-64 | grep movap | count 2

define <4 x float> @foo(<4 x float>* %p) nounwind {
  %t = load <4 x float>, <4 x float>* %p
  ret <4 x float> %t
}
define <2 x double> @bar(<2 x double>* %p) nounwind {
  %t = load <2 x double>, <2 x double>* %p
  ret <2 x double> %t
}
