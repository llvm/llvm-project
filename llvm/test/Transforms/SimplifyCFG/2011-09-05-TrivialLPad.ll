; RUN: opt < %s -simplifycfg -S | FileCheck %s

; CHECK-NOT: invoke
; CHECK-NOT: landingpad

declare void @bar()

define i32 @foo() personality i32 (i32, i64, i8*, i8*)* @__gxx_personality_v0 {
entry:
  invoke void @bar()
          to label %return unwind label %lpad

return:
  ret i32 0

lpad:
  %lp = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %lp
}

declare i32 @__gxx_personality_v0(i32, i64, i8*, i8*)
