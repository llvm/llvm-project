;; RUN: opt -passes=rewrite-statepoints-for-gc,verify -S < %s | FileCheck %s
;; This test is to verify that RewriteStatepointsForGC correctly relocates values
;; defined by invoke instruction results.

declare ptr addrspace(1) @non_gc_call() "gc-leaf-function"

declare void @gc_call()

declare ptr @fake_personality_function()

define ptr addrspace(1) @test() gc "statepoint-example" personality ptr @fake_personality_function {
; CHECK-LABEL: @test(

entry:
  %obj = invoke ptr addrspace(1) @non_gc_call()
          to label %normal_dest unwind label %unwind_dest

unwind_dest:                                      ; preds = %entry
  %lpad = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef

normal_dest:                                      ; preds = %entry
; CHECK: normal_dest:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc ptr addrspace(1)

  call void @gc_call() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %obj
}
