; RUN: llc -mtriple=armv7-eabi -mcpu=cortex-a8 < %s
; PR5410

%0 = type { float, float, float, float }
%pln = type { %vec, float }
%vec = type { [4 x float] }

define arm_aapcs_vfpcc float @aaa(%vec* nocapture %ustart, %vec* nocapture %udir, %vec* nocapture %vstart, %vec* nocapture %vdir, %vec* %upoint, %vec* %vpoint) {
entry:
  br i1 undef, label %bb81, label %bb48

bb48:                                             ; preds = %entry
  %0 = call arm_aapcs_vfpcc  %0 @bbb(%pln* undef, %vec* %vstart, %vec* undef) nounwind ; <%0> [#uses=0]
  ret float 0.000000e+00

bb81:                                             ; preds = %entry
  ret float 0.000000e+00
}

declare arm_aapcs_vfpcc %0 @bbb(%pln* nocapture, %vec* nocapture, %vec* nocapture) nounwind
