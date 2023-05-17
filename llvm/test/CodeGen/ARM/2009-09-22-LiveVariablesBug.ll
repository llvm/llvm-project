; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mattr=+neon

; PR5024

%bar = type { %foo, %foo }
%foo = type { <4 x float> }

declare arm_aapcs_vfpcc float @aaa(ptr nocapture) nounwind readonly

declare arm_aapcs_vfpcc ptr @bbb(ptr, <4 x float>, <4 x float>) nounwind

define arm_aapcs_vfpcc void @ccc(ptr nocapture %pBuffer, i32 %numItems) nounwind {
entry:
  br i1 undef, label %return, label %bb.nph

bb.nph:                                           ; preds = %entry
  %0 = call arm_aapcs_vfpcc  ptr @bbb(ptr undef, <4 x float> undef, <4 x float> undef) nounwind ; <ptr> [#uses=0]
  %1 = call arm_aapcs_vfpcc  float @aaa(ptr undef) nounwind ; <float> [#uses=0]
  unreachable

return:                                           ; preds = %entry
  ret void
}
