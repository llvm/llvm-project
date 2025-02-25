;PR15293: ARM codegen ice - expected larger existing stack allocation
;RUN: llc -mtriple=arm-linux-gnueabihf < %s | FileCheck %s

%struct.S227 = type { [49 x i32], i32 }

define void @check227(
                      i32 %b,
                      ptr byval(%struct.S227) nocapture %arg0,
                      ptr %arg1) {
; b --> R0
; arg0 --> [R1, R2, R3, SP+0 .. SP+188)
; arg1 --> SP+188

entry:
; CHECK: sub     sp, sp, #12
; CHECK: stm     sp, {r1, r2, r3}
; CHECK: ldr     r0, [sp, #200]
; CHECK: add     sp, sp, #12
; CHECK: b       useInt

  %0 = ptrtoint ptr %arg1 to i32
  tail call void @useInt(i32 %0)
  ret void
}

declare void @useInt(i32)

