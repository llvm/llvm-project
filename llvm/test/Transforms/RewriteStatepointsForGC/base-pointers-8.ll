; RUN: opt %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s

; CHECK: derived %next_element_ptr base %array_obj

define i32 @null_in_array(i64 addrspace(1)* %array_obj) gc "statepoint-example" {
entry:
  %array_len_pointer.i64 = getelementptr i64, i64 addrspace(1)* %array_obj, i32 1
  %array_len_pointer.i32 = bitcast i64 addrspace(1)* %array_len_pointer.i64 to i32 addrspace(1)*
  %array_len = load i32, i32 addrspace(1)* %array_len_pointer.i32
  %array_elems = bitcast i32 addrspace(1)* %array_len_pointer.i32 to i64 addrspace(1)* addrspace(1)*
  br label %loop_check

loop_check:
  %index = phi i32 [ 0, %entry ], [ %next_index, %loop_back ]
  %current_element_ptr = phi i64 addrspace(1)* addrspace(1)* [ %array_elems, %entry ], [ %next_element_ptr, %loop_back ]
  %index_lt = icmp ult i32 %index, %array_len
  br i1 %index_lt, label %check_for_null, label %not_found

check_for_null:
  %current_element = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(1)* %current_element_ptr
  %is_null = icmp eq i64 addrspace(1)* %current_element, null
  br i1 %is_null, label %found, label %loop_back

loop_back:
  %next_element_ptr = getelementptr i64 addrspace(1)*, i64 addrspace(1)* addrspace(1)* %current_element_ptr, i32 1
  %next_index = add i32 %index, 1
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %loop_check

not_found:
  ret i32 -1

found:
  ret i32 %index
}

declare void @do_safepoint()
declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)