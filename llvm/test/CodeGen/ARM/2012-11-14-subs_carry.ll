; RUN: llc < %s -mtriple=thumbv7-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s

;CHECK-LABEL: foo:
;CHECK: movs r0, #0
;CHECK-NEXT: bx lr

; Note: This test case originally checked, per r167963, for:
;       adds
;       adc
;       bx
; But SDAG now, like InstCombine, can fold everything away.
;rdar://12028498

define i32 @foo() nounwind ssp {
entry:
  %tmp2 = zext i32 3 to i64
  br  label %bug_block

bug_block:
  %tmp410 = and i64 1031, 1647010
  %tmp411 = and i64 %tmp2, -211
  %tmp412 = shl i64 %tmp410, %tmp2
  %tmp413 = shl i64 %tmp411, %tmp2
  %tmp415 = and i64 %tmp413, 1
  %tmp420 = xor i64 0, %tmp415
  %tmp421 = and i64 %tmp412, %tmp415
  %tmp422 = shl i64 %tmp421, 1
  br  label %finish

finish:
  %tmp423 = lshr i64 %tmp422, 32
  %tmp424 = trunc i64 %tmp423 to i32
  ret i32 %tmp424
}

