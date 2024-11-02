; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; CHECK-NOT: invoke
; CHECK-NOT: landingpad

declare void @bar()

define i32 @foo() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @bar()
          to label %return unwind label %lpad

return:
  ret i32 0

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp
}

declare i32 @__gxx_personality_v0(i32, i64, ptr, ptr)
