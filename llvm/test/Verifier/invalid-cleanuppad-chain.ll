; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

declare i32 @__gxx_personality_v0(...)

; CHECK: CleanupReturnInst needs to be provided a CleanupPad
; CHECK-NEXT: cleanupret from undef unwind label %bb2
; CHECK-NEXT: token undef
; CHECK: Parent pad must be catchpad/cleanuppad/catchswitch
; CHECK-NEXT: cleanupret from undef unwind label %bb2

define void @test() personality ptr @__gxx_personality_v0 {
  br label %bb1

bb1:
  cleanupret from undef unwind label %bb2

bb2:
  %pad = cleanuppad within none []
  cleanupret from %pad unwind to caller
}
