; RUN: opt < %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S  2>&1 | FileCheck %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -spp-print-base-pointers -S  2>&1 | FileCheck %s

; CHECK: derived %next base %base_obj

declare i1 @runtime_value() "gc-leaf-function"

define void @maybe_GEP(i64 addrspace(1)* %base_obj) gc "statepoint-example" {
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %current = phi i64 addrspace(1)* [ %base_obj, %entry ], [ %next, %loop ]
  %condition = call i1 @runtime_value()
  %maybe_next = getelementptr i64, i64 addrspace(1)* %current, i32 1
  %next = select i1 %condition, i64 addrspace(1)* %maybe_next, i64 addrspace(1)* %current
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  br label %loop
}

declare void @do_safepoint()
