; RUN: opt -S -passes=spirv-prepare-functions -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

declare i32 @__gxx_personality_v0(...)

; invoke call sites must get their FunctionType patched too, like plain calls.
; CHECK-LABEL: define void @invoke_caller(
; CHECK: invoke i32 @callback(i32 %x)
define void @invoke_caller({ float, float } %x) personality ptr @__gxx_personality_v0 {
entry:
  %r = invoke { float, float } @callback({ float, float } %x)
          to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  ret void
}

; CHECK-LABEL: define i32 @callback(
define { float, float } @callback({ float, float } %x) {
  ret { float, float } %x
}
