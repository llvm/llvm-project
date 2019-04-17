; RUN: opt < %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s

; CHECK: derived %merged_value base %merged_value.base

declare void @site_for_call_safpeoint()

define i64 addrspace(1)* @test(i64 addrspace(1)* %base_obj_x, i64 addrspace(1)* %base_obj_y, i1 %runtime_condition) gc "statepoint-example" {
entry:
  br i1 %runtime_condition, label %here, label %there

here:                                             ; preds = %entry
  %x = getelementptr i64, i64 addrspace(1)* %base_obj_x, i32 1
  br label %merge

there:                                            ; preds = %entry
  %y = getelementptr i64, i64 addrspace(1)* %base_obj_y, i32 1
  br label %merge

merge:                                            ; preds = %there, %here
; CHECK-LABEL: merge:
; CHECK:   %merged_value.base = phi i64 addrspace(1)* [ %base_obj_x, %here ], [ %base_obj_y, %there ]
  %merged_value = phi i64 addrspace(1)* [ %x, %here ], [ %y, %there ]
  call void @site_for_call_safpeoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i64 addrspace(1)* %merged_value
}
