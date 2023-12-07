; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=function-data --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

; INTERESTING: define void @drop_personality(
; RESULT: define void @drop_personality() {
define void @drop_personality() personality ptr @__gxx_personality_v0 {
  ret void
}

; CHECK: define void @keep_personality() personality ptr @__gxx_personality_v0 {
define void @keep_personality() personality ptr @__gxx_personality_v0 {
  ret void
}

; Make sure an invalid reduction isn't produced if we need a
; personality function for different instructions

; CHECK: define void @landingpad_requires_personality()
; RESULT-SAME: personality ptr @__gxx_personality_v0 {
define void @landingpad_requires_personality() personality ptr @__gxx_personality_v0 {
bb0:
  br label %bb2

bb1:
  landingpad { ptr, i32 }
  catch ptr null
  ret void

bb2:
  ret void
}

; CHECK-LABEL: define void @uses_catchpad()
; RESULT-SAME: personality ptr @__CxxFrameHandler3 {
define void @uses_catchpad() personality ptr @__CxxFrameHandler3 {
entry:
  br label %unreachable

catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  br label %unreachable

unreachable:
  unreachable
}

; CHECK-LABEL: define void @uses_resume()
; RESULT-SAME: personality ptr @__gxx_personality_v0
define void @uses_resume() personality ptr @__gxx_personality_v0 {
entry:
  resume { ptr, i32 } zeroinitializer
}

declare i32 @__gxx_personality_v0(...)
declare i32 @__CxxFrameHandler3(...)
