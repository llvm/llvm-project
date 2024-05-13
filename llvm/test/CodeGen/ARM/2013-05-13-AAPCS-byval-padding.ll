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

;CHECK:  sub   sp, sp, #12
;CHECK:  push  {r11, lr}
;CHECK:  sub   sp, sp, #4
;CHECK:  add   r0, sp, #12
;CHECK:  stm   r0, {r1, r2, r3}
;CHECK:  ldr   r0, [sp, #212]
;CHECK:  bl    useInt
;CHECK:  add   sp, sp, #4
;CHECK:  pop   {r11, lr}
;CHECK:  add   sp, sp, #12

  %0 = ptrtoint ptr %arg1 to i32
  tail call void @useInt(i32 %0)
  ret void
}

declare void @useInt(i32)

