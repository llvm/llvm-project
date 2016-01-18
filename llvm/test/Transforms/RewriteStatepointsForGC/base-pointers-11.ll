; RUN: opt %s -rewrite-statepoints-for-gc -spp-print-base-pointers  -S 2>&1 | FileCheck %s

; CHECK: derived %next base %base_obj

declare void @do_safepoint()

define void @test(i64 addrspace(1)* %base_obj) gc "statepoint-example" {
entry:
  %obj = getelementptr i64, i64 addrspace(1)* %base_obj, i32 1
  br label %loop

loop:                                             ; preds = %loop, %entry
; CHECK-LABEL: loop:
; CHECK: phi i64 addrspace(1)*
; CHECK-DAG:  [ %base_obj.relocated.casted, %loop ] 
; CHECK-DAG:  [ %base_obj, %entry ]
; CHECK:  %current = phi i64 addrspace(1)* 
; CHECK-DAG:  [ %obj, %entry ]
; CHECK-DAG:  [ %next.relocated.casted, %loop ]
  %current = phi i64 addrspace(1)* [ %obj, %entry ], [ %next, %loop ]
  %next = getelementptr i64, i64 addrspace(1)* %current, i32 1
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %loop
}

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)