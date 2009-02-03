; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2
; PR2762
define void @foo(<4 x i32>* %p, <4 x double>* %q) {
  %n = load <4 x i32>* %p
  %z = sitofp <4 x i32> %n to <4 x double>
  store <4 x double> %z, <4 x double>* %q
  ret void
}
