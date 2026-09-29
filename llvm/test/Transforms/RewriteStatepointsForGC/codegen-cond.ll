; RUN: opt -passes=rewrite-statepoints-for-gc -S < %s | FileCheck %s

; A null test of a single value

define i1 @test(ptr addrspace(1) %p, i1 %rare) gc "statepoint-example" {
; CHECK-LABEL: @test
entry:
  %cond = icmp eq ptr addrspace(1) %p, null
  br i1 %rare, label %safepoint, label %continue, !prof !0

safepoint:                                        ; preds = %entry
  call void @safepoint() [ "deopt"() ]
  br label %continue

continue:                                         ; preds = %safepoint, %entry
; CHECK-LABEL: continue:
; CHECK: phi
; CHECK-DAG: [ %p.relocated, %safepoint ]
; CHECK-DAG: [ %p, %entry ]
; CHECK: %cond = icmp
; CHECK: br i1 %cond
; Comparing two pointers
  br i1 %cond, label %taken, label %untaken

taken:                                            ; preds = %continue
  ret i1 true

untaken:                                          ; preds = %continue
  ret i1 false
}

define i1 @test2(ptr addrspace(1) %p, ptr addrspace(1) %q, i1 %rare) gc "statepoint-example" {
; CHECK-LABEL: @test2
entry:
  %cond = icmp eq ptr addrspace(1) %p, %q
  br i1 %rare, label %safepoint, label %continue, !prof !0

safepoint:                                        ; preds = %entry
  call void @safepoint() [ "deopt"() ]
  br label %continue

continue:                                         ; preds = %safepoint, %entry
; CHECK-LABEL: continue:
; CHECK: phi
; CHECK-DAG: [ %q.relocated, %safepoint ]
; CHECK-DAG: [ %q, %entry ]
; CHECK: phi
; CHECK-DAG: [ %p.relocated, %safepoint ]
; CHECK-DAG: [ %p, %entry ]
; CHECK: %cond = icmp
; CHECK: br i1 %cond
; Check that nothing bad happens if already last instruction
; before terminator
  br i1 %cond, label %taken, label %untaken

taken:                                            ; preds = %continue
  ret i1 true

untaken:                                          ; preds = %continue
  ret i1 false
}

define i1 @test3(ptr addrspace(1) %p, ptr addrspace(1) %q, i1 %rare) gc "statepoint-example" {
; CHECK-LABEL: @test3
; CHECK: gc.statepoint
; CHECK: %cond = icmp
; CHECK: br i1 %cond
entry:
  call void @safepoint() [ "deopt"() ]
  %cond = icmp eq ptr addrspace(1) %p, %q
  br i1 %cond, label %taken, label %untaken

taken:                                            ; preds = %entry
  ret i1 true

untaken:                                          ; preds = %entry
  ret i1 false
}

declare void @safepoint()
!0 = !{!"branch_weights", i32 1, i32 10000}
