; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep cvttpd2pi | count 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep cvtpi2pd | count 1
; PR2687

define <2 x double> @a(<2 x i32> %x) nounwind {
entry:
  %y = sitofp <2 x i32> %x to <2 x double>
  ret <2 x double> %y
}

define <2 x i32> @b(<2 x double> %x) nounwind {
entry:
  %y = fptosi <2 x double> %x to <2 x i32>
  ret <2 x i32> %y
}
