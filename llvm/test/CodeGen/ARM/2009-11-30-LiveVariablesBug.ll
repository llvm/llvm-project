; RUN: llc -mtriple=armv7-eabi -mcpu=cortex-a8 < %s
; PR5614

%"als" = type { i32 (...)** }
%"av" = type { %"als" }
%"c" = type { %"lsm", %"Vec3", %"av"*, float, i8, float, %"lsm", i8, %"Vec3", %"Vec3", %"Vec3", float, float, float, %"Vec3", %"Vec3" }
%"lsm" = type { %"als", %"Vec3", %"Vec3", %"Vec3", %"Vec3" }
%"Vec3" = type { float, float, float }

define arm_aapcs_vfpcc void @foo(%"c"* %this, %"Vec3"* nocapture %adjustment) {
entry:
  switch i32 undef, label %return [
    i32 1, label %bb
    i32 2, label %bb72
    i32 3, label %bb31
    i32 4, label %bb79
    i32 5, label %bb104
  ]

bb:                                               ; preds = %entry
  ret void

bb31:                                             ; preds = %entry
  %0 = call arm_aapcs_vfpcc  %"Vec3" undef(%"lsm"* undef) ; <%"Vec3"> [#uses=1]
  %mrv_gr69 = extractvalue %"Vec3" %0, 1 ; <float> [#uses=1]
  %1 = fsub float %mrv_gr69, undef                ; <float> [#uses=1]
  store float %1, float* undef, align 4
  ret void

bb72:                                             ; preds = %entry
  ret void

bb79:                                             ; preds = %entry
  ret void

bb104:                                            ; preds = %entry
  ret void

return:                                           ; preds = %entry
  ret void
}
