; RUN: opt < %s -passes=rewrite-statepoints-for-gc -S | FileCheck %s

declare void @some_call(ptr addrspace(1))

declare i32 @dummy_personality_function()

define ptr addrspace(1) @test(ptr addrspace(1) %obj, ptr addrspace(1) %obj1)
  gc "statepoint-example"
  personality ptr @dummy_personality_function {
entry:
  invoke void @some_call(ptr addrspace(1) %obj) [ "deopt"() ]
          to label %second_invoke unwind label %exceptional_return

second_invoke:                                    ; preds = %entry
  invoke void @some_call(ptr addrspace(1) %obj) [ "deopt"() ]
          to label %normal_return unwind label %exceptional_return

normal_return:                                    ; preds = %second_invoke
  ret ptr addrspace(1) %obj

; CHECK: exceptional_return1:
; CHECK-NEXT: %lpad2 = landingpad token

; CHECK: exceptional_return.split-lp:
; CHECK-NEXT: %lpad.split-lp = landingpad token

; CHECK: exceptional_return:
; CHECK-NOT: phi token

exceptional_return:                               ; preds = %second_invoke, %entry
  %lpad = landingpad token cleanup
  ret ptr addrspace(1) %obj1
}
