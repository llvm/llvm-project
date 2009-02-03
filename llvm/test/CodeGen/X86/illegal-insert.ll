; RUN: llvm-as %s -o - | llc -march=x86-64

define <4 x double> @foo0(<4 x double> %t) {
  %r = insertelement <4 x double> %t, double 2.3, i32 0
  ret <4 x double> %r
}
define <4 x double> @foo1(<4 x double> %t) {
  %r = insertelement <4 x double> %t, double 2.3, i32 1
  ret <4 x double> %r
}
define <4 x double> @foo2(<4 x double> %t) {
  %r = insertelement <4 x double> %t, double 2.3, i32 2
  ret <4 x double> %r
}
define <4 x double> @foo3(<4 x double> %t) {
  %r = insertelement <4 x double> %t, double 2.3, i32 3
  ret <4 x double> %r
}
