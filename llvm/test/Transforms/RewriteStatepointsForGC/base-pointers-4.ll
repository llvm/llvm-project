; RUN: opt < %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s

; CHECK: derived %obj_to_consume base %obj_to_consume.base

declare void @foo()

declare i64 addrspace(1)* @generate_obj()

declare void @consume_obj(i64 addrspace(1)*)

define void @test(i32 %condition) gc "statepoint-example" {
entry:
  br label %loop

loop:                                             ; preds = %merge.split, %entry
; CHECK: loop:
; CHECK:  [[TOKEN_0:%[^ ]+]] = call token (i64, i32, i64 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i64f(i64 2882400000, i32 0, i64 addrspace(1)* ()* @generate_obj, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i3
; CHECK-NEXT:  [[RESULT_0:%[^ ]+]] = call i64 addrspace(1)* @llvm.experimental.gc.result
  %0 = call i64 addrspace(1)* @generate_obj() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  switch i32 %condition, label %dest_a [
    i32 0, label %dest_b
    i32 1, label %dest_c
  ]

dest_a:                                           ; preds = %loop
  br label %merge

dest_b:                                           ; preds = %loop
  br label %merge

dest_c:                                           ; preds = %loop
  br label %merge

merge:                                            ; preds = %dest_c, %dest_b, %dest_a
; CHECK: merge:
; CHECK:  %obj_to_consume = phi i64 addrspace(1)* [ [[RESULT_0]], %dest_a ], [ null, %dest_b ], [ null, %dest_c ]
  %obj_to_consume = phi i64 addrspace(1)* [ %0, %dest_a ], [ null, %dest_b ], [ null, %dest_c ]
  call void @consume_obj(i64 addrspace(1)* %obj_to_consume) [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  br label %merge.split

merge.split:                                      ; preds = %merge
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  br label %loop
}
