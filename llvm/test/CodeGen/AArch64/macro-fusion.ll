; REQUIRES: asserts
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+fuse-arith-logic -verify-misched -debug-only=machine-scheduler 2>&1 > /dev/null | FileCheck %s

; Verify that, the macro-fusion creates the necessary dependencies between SUs and
; only 2 SU's are fused at most.
define signext i32 @test(i32 signext %a, i32 signext %b, i32 signext %c, i32 signext %d) {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: %bb.0 entry 
; CHECK: Macro fuse: SU([[SU4:[0-9]+]]) - SU([[SU5:[0-9]+]])
; CHECK: Bind SU([[SU1:[0-9]+]]) - SU([[SU4]])
; CHECK-NOT: Macro fuse:
; CHECK: SU([[SU1]]):   %{{[0-9]+}}:gpr32 = COPY $w2
; CHECK: SU([[SU4]]):   %{{[0-9]+}}:gpr32 = nsw ADDWrr
; CHECK: SU([[SU5]]):   %{{[0-9]+}}:gpr32 = nsw ADDWrr
  %add = add nsw i32 %b, %a
  %add1 = add nsw i32 %add, %c
  %sub = sub nsw i32 %add1, %d
  ret i32 %sub
}
