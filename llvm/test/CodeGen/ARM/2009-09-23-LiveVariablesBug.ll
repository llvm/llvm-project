; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mattr=+neon

; PR5024

%struct.1 = type { %struct.4, %struct.4 }
%struct.4 = type { <4 x float> }

define arm_aapcs_vfpcc ptr @hhh3(ptr %this, <4 x float> %lenation.0, <4 x float> %legalation.0) nounwind {
entry:
  %0 = call arm_aapcs_vfpcc  ptr @sss1(ptr undef, float 0.000000e+00) nounwind ; <ptr> [#uses=0]
  %1 = call arm_aapcs_vfpcc  ptr @qqq1(ptr null, float 5.000000e-01) nounwind ; <ptr> [#uses=0]
  %val92 = load <4 x float>, ptr null                 ; <<4 x float>> [#uses=1]
  %2 = call arm_aapcs_vfpcc  ptr @zzz2(ptr undef, <4 x float> %val92) nounwind ; <ptr> [#uses=0]
  ret ptr %this
}

declare arm_aapcs_vfpcc ptr @qqq1(ptr, float) nounwind

declare arm_aapcs_vfpcc ptr @sss1(ptr, float) nounwind

declare arm_aapcs_vfpcc ptr @zzz2(ptr, <4 x float>) nounwind
