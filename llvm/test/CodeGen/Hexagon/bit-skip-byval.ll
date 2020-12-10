; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Either and or zxtb.
; CHECK: r0 = and(r1,#255)

%struct.t0 = type { i32 }

define i32 @foo(%struct.t0* byval(%struct.t0) align 8 %s, i8 zeroext %t, i8 %u) #0 {
  %a = zext i8 %u to i32
  ret i32 %a
}
