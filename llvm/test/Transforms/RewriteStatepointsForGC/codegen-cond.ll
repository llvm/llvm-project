; RUN: opt -rewrite-statepoints-for-gc -S < %s | FileCheck %s

; A null test of a single value
define i1 @test(i8 addrspace(1)* %p, i1 %rare) gc "statepoint-example" {
; CHECK-LABEL: @test
entry:
   %cond = icmp eq i8 addrspace(1)* %p, null
   br i1 %rare, label %safepoint, label %continue, !prof !0
safepoint:
   call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @safepoint, i32 0, i32 0, i32 0, i32 0)
   br label %continue
continue:
; CHECK-LABEL: continue:
; CHECK: phi
; CHECK-DAG: [ %p.relocated, %safepoint ]
; CHECK-DAG: [ %p, %entry ]
; CHECK: %cond = icmp
; CHECK: br i1 %cond
   br i1 %cond, label %taken, label %untaken
taken:
   ret i1 true
untaken:
   ret i1 false
}

; Comparing two pointers
define i1 @test2(i8 addrspace(1)* %p, i8 addrspace(1)* %q, i1 %rare) 
    gc "statepoint-example" {
; CHECK-LABEL: @test2
entry:   
   %cond = icmp eq i8 addrspace(1)* %p, %q
   br i1 %rare, label %safepoint, label %continue, !prof !0
safepoint:
   call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @safepoint, i32 0, i32 0, i32 0, i32 0)
   br label %continue
continue:
; CHECK-LABEL: continue:
; CHECK: phi
; CHECK-DAG: [ %q.relocated, %safepoint ]
; CHECK-DAG: [ %q, %entry ]
; CHECK: phi
; CHECK-DAG: [ %p.relocated, %safepoint ]
; CHECK-DAG: [ %p, %entry ]
; CHECK: %cond = icmp
; CHECK: br i1 %cond
   br i1 %cond, label %taken, label %untaken
taken:
   ret i1 true
untaken:
   ret i1 false
}

; Sanity check that nothing bad happens if already last instruction
; before terminator
define i1 @test3(i8 addrspace(1)* %p, i8 addrspace(1)* %q, i1 %rare) 
    gc "statepoint-example" {
; CHECK-LABEL: @test3
entry:   
   call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @safepoint, i32 0, i32 0, i32 0, i32 0)
; CHECK: gc.statepoint
; CHECK: %cond = icmp
; CHECK: br i1 %cond
   %cond = icmp eq i8 addrspace(1)* %p, %q
   br i1 %cond, label %taken, label %untaken
taken:
   ret i1 true
untaken:
   ret i1 false
}

declare void @safepoint()
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

!0 = !{!"branch_weights", i32 1, i32 10000}
