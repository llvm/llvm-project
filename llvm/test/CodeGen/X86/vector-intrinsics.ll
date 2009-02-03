; RUN: llvm-as < %s | llc -march=x86-64 | grep call | count 16

declare <4 x double> @llvm.sin.v4f64(<4 x double> %p)
declare <4 x double> @llvm.cos.v4f64(<4 x double> %p)
declare <4 x double> @llvm.pow.v4f64(<4 x double> %p, <4 x double> %q)
declare <4 x double> @llvm.powi.v4f64(<4 x double> %p, i32)

define <4 x double> @foo(<4 x double> %p)
{
  %t = call <4 x double> @llvm.sin.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
define <4 x double> @goo(<4 x double> %p)
{
  %t = call <4 x double> @llvm.cos.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
define <4 x double> @moo(<4 x double> %p, <4 x double> %q)
{
  %t = call <4 x double> @llvm.pow.v4f64(<4 x double> %p, <4 x double> %q)
  ret <4 x double> %t
}
define <4 x double> @zoo(<4 x double> %p, i32 %q)
{
  %t = call <4 x double> @llvm.powi.v4f64(<4 x double> %p, i32 %q)
  ret <4 x double> %t
}
