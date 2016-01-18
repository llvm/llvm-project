; RUN: opt %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s

; CHECK: derived %next.i64 base %base_obj

define void @test(i64 addrspace(1)* %base_obj) gc "statepoint-example" {
entry:
  %obj = getelementptr i64, i64 addrspace(1)* %base_obj, i32 1
  br label %loop

loop:
  %current = phi i64 addrspace(1)* [ %obj, %entry ], [ %next.i64, %loop ]
  %current.i32 = bitcast i64 addrspace(1)* %current to i32 addrspace(1)*
  %next.i32 = getelementptr i32, i32 addrspace(1)* %current.i32, i32 1
  %next.i64 = bitcast i32 addrspace(1)* %next.i32 to i64 addrspace(1)*
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %loop
}

declare void @do_safepoint()
declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)