; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mattr=+neon

; PR5024

%struct.1 = type { %struct.4, %struct.4 }
%struct.4 = type { <4 x float> }

define arm_aapcs_vfpcc %struct.1* @hhh3(%struct.1* %this, <4 x float> %lenation.0, <4 x float> %legalation.0) nounwind {
entry:
  %0 = call arm_aapcs_vfpcc  %struct.4* @sss1(%struct.4* undef, float 0.000000e+00) nounwind ; <%struct.4*> [#uses=0]
  %1 = call arm_aapcs_vfpcc  %struct.4* @qqq1(%struct.4* null, float 5.000000e-01) nounwind ; <%struct.4*> [#uses=0]
  %val92 = load <4 x float>* null                 ; <<4 x float>> [#uses=1]
  %2 = call arm_aapcs_vfpcc  %struct.4* @zzz2(%struct.4* undef, <4 x float> %val92) nounwind ; <%struct.4*> [#uses=0]
  ret %struct.1* %this
}

declare arm_aapcs_vfpcc %struct.4* @qqq1(%struct.4*, float) nounwind

declare arm_aapcs_vfpcc %struct.4* @sss1(%struct.4*, float) nounwind

declare arm_aapcs_vfpcc %struct.4* @zzz2(%struct.4*, <4 x float>) nounwind
