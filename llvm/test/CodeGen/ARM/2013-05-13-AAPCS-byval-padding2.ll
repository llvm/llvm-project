;PR15293: ARM codegen ice - expected larger existing stack allocation
;RUN: llc -mtriple=arm-linux-gnueabihf < %s | FileCheck %s

%struct4bytes = type { i32 }
%struct20bytes = type { i32, i32, i32, i32, i32 }

define void @foo(ptr byval(%struct4bytes) %p0, ; --> R0
                 ptr byval(%struct20bytes) %p1 ; --> R1,R2,R3, [SP+0 .. SP+8)
) {
;CHECK:  sub  sp, sp, #16
;CHECK:  push  {r11, lr}
;CHECK:  add  r12, sp, #8
;CHECK:  stm  r12, {r0, r1, r2, r3}
;CHECK:  add  r0, sp, #12
;CHECK:  bl  useInt
;CHECK:  pop  {r11, lr}
;CHECK:  add  sp, sp, #16

  %1 = ptrtoint ptr %p1 to i32
  tail call void @useInt(i32 %1)
  ret void
}

declare void @useInt(i32)

