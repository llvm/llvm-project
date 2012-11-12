; RUN: llc < %s -march=x86-64 | grep call | count 43

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


declare <9 x double> @llvm.exp.v9f64(<9 x double> %a)
declare <9 x double> @llvm.pow.v9f64(<9 x double> %a, <9 x double> %b)
declare <9 x double> @llvm.powi.v9f64(<9 x double> %a, i32)

define void @a(<9 x double>* %p) nounwind {
  %a = load <9 x double>* %p
  %r = call <9 x double> @llvm.exp.v9f64(<9 x double> %a)
  store <9 x double> %r, <9 x double>* %p
  ret void
}
define void @b(<9 x double>* %p, <9 x double>* %q) nounwind {
  %a = load <9 x double>* %p
  %b = load <9 x double>* %q
  %r = call <9 x double> @llvm.pow.v9f64(<9 x double> %a, <9 x double> %b)
  store <9 x double> %r, <9 x double>* %p
  ret void
}
define void @c(<9 x double>* %p, i32 %n) nounwind {
  %a = load <9 x double>* %p
  %r = call <9 x double> @llvm.powi.v9f64(<9 x double> %a, i32 %n)
  store <9 x double> %r, <9 x double>* %p
  ret void
}
