; RUN: opt %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s

; CHECK: derived %merged_value base %base_obj


define i64 addrspace(1)* @test(i64 addrspace(1)* %base_obj, i1 %runtime_condition) gc "statepoint-example" {
entry:
  br i1 %runtime_condition, label %merge, label %there

there:
  %derived_obj = getelementptr i64, i64 addrspace(1)* %base_obj, i32 1
  br label %merge

merge:
  %merged_value = phi i64 addrspace(1)* [ %base_obj, %entry ], [ %derived_obj, %there ]
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %merged_value
}

declare void @foo()
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)