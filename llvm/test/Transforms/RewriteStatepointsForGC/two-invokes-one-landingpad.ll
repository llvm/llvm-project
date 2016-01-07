; RUN: opt %s -rewrite-statepoints-for-gc -rs4gc-use-deopt-bundles -S | FileCheck %s

declare void @some_call(i64 addrspace(1)*)

declare i32 @"dummy_personality_function"()

define i64 addrspace(1)* @test(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1)
  gc "statepoint-example"
  personality i32 ()* @"dummy_personality_function" {
entry:
  invoke void @some_call(i64 addrspace(1)* %obj) [ "deopt"() ]
          to label %second_invoke unwind label %exceptional_return

second_invoke:                                    ; preds = %entry
  invoke void @some_call(i64 addrspace(1)* %obj) [ "deopt"() ]
          to label %normal_return unwind label %exceptional_return

normal_return:                                    ; preds = %second_invoke
  ret i64 addrspace(1)* %obj

; CHECK: exceptional_return1:
; CHECK-NEXT: %lpad2 = landingpad token

; CHECK: exceptional_return.split-lp:
; CHECK-NEXT: %lpad.split-lp = landingpad token

; CHECK: exceptional_return:
; CHECK-NOT: phi token

exceptional_return:                               ; preds = %second_invoke, %entry
  %lpad = landingpad token cleanup
  ret i64 addrspace(1)* %obj1
}
