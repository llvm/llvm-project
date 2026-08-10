; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=ppc
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=ppc64

define <4 x float> @foo1(<2 x float> %a, <2 x float> %b) nounwind readnone {
entry:
  %0 = shufflevector <2 x float> %a, <2 x float> undef, <4 x i32> <i32 0, i32 undef, i32 1, i32 undef>
  %1 = shufflevector <2 x float> %b, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %2 = shufflevector <4 x float> %0, <4 x float> %1, <4 x i32> <i32 0, i32 4, i32 2, i32 5>
  ret <4 x float> %2
}

define <4 x double> @foo2(<2 x double> %a, <2 x double> %b) nounwind readnone {
entry:
  %0 = shufflevector <2 x double> %a, <2 x double> undef, <4 x i32> <i32 0, i32 undef, i32 1, i32 undef>
  %1 = shufflevector <2 x double> %b, <2 x double> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %2 = shufflevector <4 x double> %0, <4 x double> %1, <4 x i32> <i32 0, i32 4, i32 2, i32 5>
  ret <4 x double> %2
}

