; RUN: opt %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s


; CHECK: derived %obj_to_consume base %obj_to_consume

declare void @foo()
declare i64 addrspace(1)* @generate_obj()
declare void @consume_obj(i64 addrspace(1)*)

define void @test(i32 %condition) gc "statepoint-example" {
entry:
  br label %loop

loop:
; CHECK: loop:
; CHECK:  %safepoint_token1 = call i32 (i64, i32, i64 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i64f(i64 0, i32 0, i64 addrspace(1)* ()* @generate_obj, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i3
; CHECK-NEXT:  %obj2 = call i64 addrspace(1)* @llvm.experimental.gc.result
  %safepoint_token1 = call i32 (i64, i32, i64 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i64f(i64 0, i32 0, i64 addrspace(1)* ()* @generate_obj, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  %obj2 = call i64 addrspace(1)* @llvm.experimental.gc.result.p1i64(i32 %safepoint_token1)
  switch i32 %condition, label %dest_a [
    i32 0, label %dest_b
    i32 1, label %dest_c
  ]

dest_a:
  br label %merge

dest_b:
  br label %merge

dest_c:
  br label %merge

merge:
; CHECK: merge:
; CHECK:  %obj_to_consume = phi i64 addrspace(1)* [ %obj2, %dest_a ], [ null, %dest_b ], [ null, %dest_c ]

  %obj_to_consume = phi i64 addrspace(1)* [ %obj2, %dest_a ], [ null, %dest_b ], [ null, %dest_c ]
  %safepoint_token3 = call i32 (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* @consume_obj, i32 1, i32 0, i64 addrspace(1)* %obj_to_consume, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %merge.split

merge.split:                                      ; preds = %merge
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %loop
}


; Function Attrs: nounwind
declare i64 addrspace(1)* @llvm.experimental.gc.result.p1i64(i32) #0
declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 @llvm.experimental.gc.statepoint.p0f_p1i64f(i64, i32, i64 addrspace(1)* ()*, i32, i32, ...)
declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...)
