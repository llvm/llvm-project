; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=unreachable-basic-blocks,basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s < %t


; CHECK-INTERESTINGNESS: @func(

; CHECK-INTERESTINGNESS: store
; CHECK-INTERESTINGNESS: store
; CHECK-INTERESTINGNESS: store


; CHECK: bb:
; CHECK-NEXT: br i1 %arg1, label %bb3, label %bb7

; CHECK: bb3: ; preds = %bb
; CHECK-NEXT: br i1 %arg2, label %bb4, label %bb7

; CHECK: bb4: ; preds = %bb3
; CHECK-NEXT: store i32 0, ptr addrspace(1) null, align 4
; CHECK-NEXT: br label %bb10

; CHECK: bb7: ; preds = %bb3, %bb
; CHECK-NEXT: %i = phi i1 [ false, %bb ], [ true, %bb3 ]
; CHECK-NEXT: store i32 1, ptr addrspace(1) null, align 4
; CHECK-NEXT: br label %bb10

; CHECK: bb10: ; preds = %bb7, %bb4
; CHECK-NEXT: store i32 2, ptr addrspace(1) null, align 4
; CHECK-NEXT: unreachable
define amdgpu_kernel void @func(i1 %arg, i1 %arg1, i1 %arg2) {
bb:
  br i1 %arg1, label %bb3, label %bb7

bb3:                                              ; preds = %bb
  br i1 %arg2, label %bb4, label %bb7

bb4:                                              ; preds = %bb3
  store i32 0, ptr addrspace(1) null
  br label %bb5

bb5:                                              ; preds = %bb4
  unreachable

bb6:                                              ; No predecessors!
  unreachable

bb7:                                              ; preds = %bb3, %bb
  %i = phi i1 [ false, %bb ], [ true, %bb3 ]
  store i32 1, ptr addrspace(1) null
  br i1 %arg, label %bb10, label %bb8

bb8:                                              ; preds = %bb7
  br i1 %i, label %bb9, label %bb9

bb9:                                              ; preds = %bb8, %bb8
  unreachable

bb10:                                             ; preds = %bb7
  store i32 2, ptr addrspace(1) null
  unreachable
}
