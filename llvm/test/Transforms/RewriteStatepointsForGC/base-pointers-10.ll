; RUN: opt %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s

; CHECK: Base Pairs (w/o Relocation):
; CHECK-DAG: derived %next base %next.base
; CHECK-DAG: derived %next_x base %base_obj_x
; CHECK-DAG: derived %next_y base %base_obj_y

declare i1 @runtime_value()
declare void @do_safepoint()

define void @select_of_phi(i64 addrspace(1)* %base_obj_x, i64 addrspace(1)* %base_obj_y) gc "statepoint-example" {
entry:
  br label %loop

loop:
  %current_x = phi i64 addrspace(1)* [ %base_obj_x , %entry ], [ %next_x, %merge ]
  %current_y = phi i64 addrspace(1)* [ %base_obj_y , %entry ], [ %next_y, %merge ]
  %current = phi i64 addrspace(1)* [ null , %entry ], [ %next , %merge ]

  %condition = call i1 @runtime_value()
  %next_x = getelementptr i64, i64 addrspace(1)* %current_x, i32 1
  %next_y = getelementptr i64, i64 addrspace(1)* %current_y, i32 1

  br i1 %condition, label %true, label %false

true:
  br label %merge

false:
  br label %merge

merge:
  %next = phi i64 addrspace(1)* [ %next_x, %true ], [ %next_y, %false ]
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %loop
}

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
