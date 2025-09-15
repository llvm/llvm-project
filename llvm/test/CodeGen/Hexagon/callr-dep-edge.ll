; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check that the callr and the load into r0 are not packetized together.

target triple = "hexagon"

@fp = common global ptr null, align 4

; CHECK: [[REG:r[0-9]+]] = memw
; CHECK: {
; CHECK: callr [[REG]]

; Function Attrs: nounwind
define i32 @foo() #0 {
entry:
  %0 = load ptr, ptr @fp, align 4
  %call = tail call i32 %0() #0
  ret i32 %call
}

attributes #0 = { nounwind }
