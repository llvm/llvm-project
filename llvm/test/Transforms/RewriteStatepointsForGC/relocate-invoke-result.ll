
;; RUN: opt -rewrite-statepoints-for-gc -verify -S < %s | FileCheck %s
;; This test is to verify that RewriteStatepointsForGC correctly relocates values
;; defined by invoke instruction results. 

declare i64* addrspace(1)* @non_gc_call() "gc-leaf-function"

declare void @gc_call()

declare i32* @fake_personality_function()

define i64* addrspace(1)* @test() gc "statepoint-example" personality i32* ()* @fake_personality_function {
; CHECK-LABEL: @test(

entry:
  %obj = invoke i64* addrspace(1)* @non_gc_call()
          to label %normal_dest unwind label %unwind_dest

unwind_dest:                                      ; preds = %entry
  %lpad = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef

normal_dest:                                      ; preds = %entry
; CHECK: normal_dest:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: bitcast

  call void @gc_call() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i64* addrspace(1)* %obj
}
