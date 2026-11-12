; A collection of liveness test cases to ensure we're reporting the
; correct live values at statepoints
; RUN: opt -passes=rewrite-statepoints-for-gc -spp-rematerialization-threshold=0 -S < %s | FileCheck %s

; Tests to make sure we consider %obj live in both the taken and untaken
; predeccessor of merge.

define ptr addrspace(1) @test1(i1 %cmp, ptr addrspace(1) %obj) gc "statepoint-example" {
; CHECK-LABEL: @test1
entry:
  br i1 %cmp, label %taken, label %untaken

taken:                                            ; preds = %entry
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc ptr addrspace(1)
; CHECK-NEXT: br label %merge
  call void @foo() [ "deopt"() ]
  br label %merge

untaken:                                          ; preds = %entry
; CHECK-LABEL: untaken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated2 = call coldcc ptr addrspace(1)
; CHECK-NEXT: br label %merge
  call void @foo() [ "deopt"() ]
  br label %merge

merge:                                            ; preds = %untaken, %taken
; CHECK-LABEL: merge:
; CHECK-NEXT: %.0 = phi ptr addrspace(1) [ %obj.relocated, %taken ], [ %obj.relocated2, %untaken ]
; CHECK-NEXT: ret ptr addrspace(1) %.0
; A local kill should not effect liveness in predecessor block
  ret ptr addrspace(1) %obj
}

define ptr addrspace(1) @test2(i1 %cmp, ptr %loc) gc "statepoint-example" {
; CHECK-LABEL: @test2
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT:  gc.statepoint
; CHECK-NEXT:  br
  call void @foo() [ "deopt"() ]
  br i1 %cmp, label %taken, label %untaken

taken:                                            ; preds = %entry
; CHECK-LABEL: taken:
; CHECK-NEXT:  %obj = load
; CHECK-NEXT:  gc.statepoint
; CHECK-NEXT:  gc.relocate
; CHECK-NEXT:  ret ptr addrspace(1) %obj.relocated
; A local kill should effect values live from a successor phi.  Also, we
; should only propagate liveness from a phi to the appropriate predecessors.
  %obj = load ptr addrspace(1), ptr %loc
  call void @foo() [ "deopt"() ]
  ret ptr addrspace(1) %obj

untaken:                                          ; preds = %entry
  ret ptr addrspace(1) null
}

define ptr addrspace(1) @test3(i1 %cmp, ptr %loc) gc "statepoint-example" {
; CHECK-LABEL: @test3
entry:
  br i1 %cmp, label %taken, label %untaken

taken:                                            ; preds = %entry
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj = load
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc ptr addrspace(1)
; CHECK-NEXT: br label %merge
  call void @foo() [ "deopt"() ]
  %obj = load ptr addrspace(1), ptr %loc
  call void @foo() [ "deopt"() ]
  br label %merge

untaken:                                          ; preds = %entry
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: br label %merge
; A base pointer must be live if it is needed at a later statepoint,
; even if the base pointer is otherwise unused.
  call void @foo() [ "deopt"() ]
  br label %merge

merge:                                            ; preds = %untaken, %taken
  %phi = phi ptr addrspace(1) [ %obj, %taken ], [ null, %untaken ]
  ret ptr addrspace(1) %phi
}

define ptr addrspace(1) @test4(i1 %cmp, ptr addrspace(1) %obj) gc "statepoint-example" {
; CHECK-LABEL: @test4
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT:  %derived = getelementptr
; CHECK-NEXT:  gc.statepoint
; CHECK-NEXT:  %derived.relocated =
; CHECK-NEXT:  %obj.relocated =
; CHECK-NEXT:  gc.statepoint
; CHECK-NEXT:  %derived.relocated2 =

; Note: It's legal to relocate obj again, but not strictly needed
; CHECK-NEXT:  %obj.relocated3 =
; CHECK-NEXT:  ret ptr addrspace(1) %derived.relocated2
;
; Make sure that a phi def visited during iteration is considered a kill.
; Also, liveness after base pointer analysis can change based on new uses,
; not just new defs.
  %derived = getelementptr i64, ptr addrspace(1) %obj, i64 8
  call void @foo() [ "deopt"() ]
  call void @foo() [ "deopt"() ]
  ret ptr addrspace(1) %derived
}

declare void @consume(...) readonly "gc-leaf-function"

define ptr addrspace(1) @test5(i1 %cmp, ptr addrspace(1) %obj) gc "statepoint-example" {
; CHECK-LABEL: @test5
entry:
  br i1 %cmp, label %taken, label %untaken

taken:                                            ; preds = %entry
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc ptr addrspace(1)
; CHECK-NEXT: br label %merge
  call void @foo() [ "deopt"() ]
  br label %merge

untaken:                                          ; preds = %entry
; CHECK-LABEL: untaken:
; CHECK-NEXT: br label %merge
  br label %merge

merge:                                            ; preds = %untaken, %taken
; CHECK-LABEL: merge:
; CHECK-NEXT: %.0 = phi ptr addrspace(1)
; CHECK-NEXT: %obj2a = phi
; CHECK-NEXT: @consume
; CHECK-NEXT: br label %final
  %obj2a = phi ptr addrspace(1) [ %obj, %taken ], [ null, %untaken ]
  call void (...) @consume(ptr addrspace(1) %obj2a)
  br label %final

final:                                            ; preds = %merge
; CHECK-LABEL: final:
; CHECK-NEXT: @consume
; CHECK-NEXT: ret ptr addrspace(1) %.0
  call void (...) @consume(ptr addrspace(1) %obj2a)
  ret ptr addrspace(1) %obj
}

declare void @foo()

