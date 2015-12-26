; RUN: opt %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S  2>&1 | FileCheck %s

; CHECK: derived %next base %base_obj

declare i1 @runtime_value()

define void @maybe_GEP(i64 addrspace(1)* %base_obj) gc "statepoint-example" {
entry:
  br label %loop

loop:
  %current = phi i64 addrspace(1)* [ %base_obj, %entry ], [ %next, %loop ]
  %condition = call i1 @runtime_value()
  %maybe_next = getelementptr i64, i64 addrspace(1)* %current, i32 1
  %next = select i1 %condition, i64 addrspace(1)* %maybe_next, i64 addrspace(1)* %current
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %loop
}

declare void @do_safepoint()
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)