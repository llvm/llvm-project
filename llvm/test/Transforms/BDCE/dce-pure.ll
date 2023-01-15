; RUN: opt -passes=bdce -S < %s | FileCheck %s

declare i32 @strlen(ptr) readonly nounwind willreturn

define void @test1() {
  call i32 @strlen( ptr null )
  ret void

; CHECK-LABEL: @test1
; CHECK-NOT: call
; CHECK: ret void
}

define i32 @test2() personality ptr @__gxx_personality_v0 {
  ; invoke of pure function should not be deleted!
  invoke i32 @strlen( ptr null ) readnone
                  to label %Cont unwind label %Other

Cont:           ; preds = %0
  ret i32 0

Other:          ; preds = %0
   %exn = landingpad {ptr, i32}
            cleanup
  ret i32 1

; CHECK-LABEL: @test2
; CHECK: invoke
; CHECK: ret i32 1
}

declare i32 @__gxx_personality_v0(...)

