; RUN: llc < %s -mtriple=i686-- -mattr=+sse2
; PR2762
define void @foo(ptr %p, ptr %q) {
  %n = load <4 x i32>, ptr %p
  %z = sitofp <4 x i32> %n to <4 x double>
  store <4 x double> %z, ptr %q
  ret void
}
