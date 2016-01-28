; RUN: opt < %s -rewrite-statepoints-for-gc -rs4gc-use-deopt-bundles -spp-print-base-pointers -S 2>&1 | FileCheck %s

; CHECK: derived %merged_value base %base_obj

define i64 addrspace(1)* @test(i64 addrspace(1)* %base_obj, i1 %runtime_condition) gc "statepoint-example" {
entry:
  br i1 %runtime_condition, label %merge, label %there

there:                                            ; preds = %entry
  %derived_obj = getelementptr i64, i64 addrspace(1)* %base_obj, i32 1
  br label %merge

merge:                                            ; preds = %there, %entry
  %merged_value = phi i64 addrspace(1)* [ %base_obj, %entry ], [ %derived_obj, %there ]
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i64 addrspace(1)* %merged_value
}

declare void @foo()
