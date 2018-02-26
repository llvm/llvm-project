; RUN: opt < %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s


declare i1 @runtime_value() "gc-leaf-function"

declare void @do_safepoint()

define void @select_of_phi(i64 addrspace(1)* %base_obj_x, i64 addrspace(1)* %base_obj_y) gc "statepoint-example" {
entry:
  br label %loop

loop:                                             ; preds = %merge, %entry
  %current_x = phi i64 addrspace(1)* [ %base_obj_x, %entry ], [ %next_x, %merge ]
  %current_y = phi i64 addrspace(1)* [ %base_obj_y, %entry ], [ %next_y, %merge ]
  %current = phi i64 addrspace(1)* [ null, %entry ], [ %next, %merge ]
  %condition = call i1 @runtime_value()
  %next_x = getelementptr i64, i64 addrspace(1)* %current_x, i32 1
  %next_y = getelementptr i64, i64 addrspace(1)* %current_y, i32 1
  br i1 %condition, label %true, label %false

true:                                             ; preds = %loop
  br label %merge

false:                                            ; preds = %loop
  br label %merge

merge:                                            ; preds = %false, %true
  %next = phi i64 addrspace(1)* [ %next_x, %true ], [ %next_y, %false ]
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  br label %loop
}
; CHECK: Base Pairs (w/o Relocation):
; CHECK-DAG: derived %next base %next.base
; CHECK-DAG: derived %next_x base %base_obj_x
; CHECK-DAG: derived %next_y base %base_obj_y
