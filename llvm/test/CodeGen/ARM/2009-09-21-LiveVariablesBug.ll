; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mattr=+neon

; PR5024

%bar = type { <4 x float> }
%foo = type { %bar, %bar, %bar, %bar }

declare arm_aapcs_vfpcc <4 x float> @bbb(ptr) nounwind

define arm_aapcs_vfpcc void @aaa(ptr noalias sret(%foo) %agg.result, ptr %tfrm) nounwind {
entry:
  %0 = call arm_aapcs_vfpcc  <4 x float> @bbb(ptr undef) nounwind ; <<4 x float>> [#uses=0]
  ret void
}
