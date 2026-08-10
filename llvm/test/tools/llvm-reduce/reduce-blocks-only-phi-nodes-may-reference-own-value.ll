; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=basic-blocks --test=FileCheck --test-arg=--check-prefix=CHECK-INTERESTINGNESS --test-arg=%s --test-arg=--input-file %s -o %t
; RUN: FileCheck %s < %t

; Make sure an invalid reduction isn't tried when deleting %bb5,
; causing the or in %bb6 to use its own output value.


; CHECK-INTERESTINGNESS: store i32 0
; CHECK-INTERESTINGNESS: store i32 1
; CHECK-INTERESTINGNESS: store i32 2

; CHECK: store i32 0
; CHECK-NEXT: br label %bb5

; CHECK: bb5:
; CHECK-NEXT: switch

; CHECK: bb6:
; CHECK-NEXT: %tmp = phi i32 [ %tmp7, %bb6 ]
; CHECK-NEXT: store i32 1
; CHECK-NEXT: %tmp7 = or i32 %tmp, 0
; CHECK-NEXT: br label %bb6

; CHECK-NOT: bb7
; CHECK: bb8:
; CHECK-NEXT: store i32 2,
define amdgpu_kernel void @snork(i32 %arg, i1 %arg1) {
bb:
  store i32 0, ptr addrspace(3) null
  br i1 %arg1, label %bb5, label %bb7

bb5:                                              ; preds = %bb5, %bb
  switch i32 %arg, label %bb5 [
    i32 0, label %bb8
    i32 1, label %bb6
  ]

bb6:                                              ; preds = %bb6, %bb5
  %tmp = phi i32 [ %tmp7, %bb6 ], [ 0, %bb5 ]
  store i32 1, ptr addrspace(3) null
  %tmp7 = or i32 %tmp, 0
  br label %bb6

bb7:
  store i32 3, ptr addrspace(3) null
  br label %bb8

bb8:                                              ; preds = %bb5
  store i32 2, ptr addrspace(3) null
  unreachable
}
