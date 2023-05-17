; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s < %t

; CHECK-INTERESTINGNESS: store i32 0,
; CHECK-INTERESTINGNESS: store i32 1,

; CHECK: bb:
; CHECK-NEXT: br label %bb10

; CHECK: bb10:
; CHECK-NEXT: br label %bb11

; CHECK: bb11:
; CHECK-NEXT: br label %bb12

; CHECK: bb12:
; CHECK-NEXT: switch i32 %arg, label %bb13 [
; CHECK-NEXT: i32 1, label %bb13
; CHECK-NEXT: i32 0, label %bb18
; CHECK-NEXT: ]

; CHECK: bb13:
; CHECK-NEXT: br label %bb17

; CHECK: bb17:
; CHECK-NEXT: store i32 0
; CHECK-NEXT: br label %bb17

; CHECK: bb18:
; CHECK-NEXT: store i32 1
; CHECK-NEXT: br label %bb18
define amdgpu_kernel void @wibble(i32 %arg, i1 %arg1, i1 %arg2) {
bb:
  br label %bb10

bb10:                                             ; preds = %bb
  br label %bb11

bb11:                                             ; preds = %bb10
  br label %bb12

bb12:                                             ; preds = %bb11
  switch i32 %arg, label %bb13 [
    i32 1, label %bb13
    i32 0, label %bb18
  ]

bb13:                                             ; preds = %bb12, %bb12
  br i1 %arg1, label %bb14, label %bb17

bb14:                                             ; preds = %bb15, %bb13
  %tmp = phi i32 [ 0, %bb15 ], [ 0, %bb13 ]
  br label %bb15

bb15:                                             ; preds = %bb14
  %tmp16 = zext i32 %tmp to i64
  br i1 %arg2, label %bb17, label %bb14

bb17:                                             ; preds = %bb17, %bb15, %bb13
  store i32 0, ptr addrspace(3) null
  br label %bb17

bb18:                                             ; preds = %bb18, %bb12
  store i32 1, ptr addrspace(3) null
  br label %bb18
}
