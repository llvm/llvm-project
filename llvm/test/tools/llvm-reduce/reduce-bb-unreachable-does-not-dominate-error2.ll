; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s < %t

; Make sure an invalid reduction isn't produced due to leaving behind
; invalid code in %bb8 after it becomes unreachable.

; CHECK-INTERESTINGNESS: store i32 0,
; CHECK-INTERESTINGNESS: store i32 1,
; CHECK-INTERESTINGNESS: store i32 2,


; CHECK: bb:
; CHECK-NEXT: store i32 0, ptr addrspace(3) null, align 4

; CHECK: bb6: ; preds = %bb8, %bb
; CHECK-NEXT: store i32 1, ptr addrspace(3) null, align 4

; CHECK: bb8: ; preds = %bb6
; CHECK-NEXT: %tmp = phi ptr addrspace(5) [ null, %bb6 ]
define amdgpu_kernel void @foo(i32 %arg) {
bb:
  store i32 0, ptr addrspace(3) null
  br label %bb6

bb6:                                              ; preds = %bb10, %bb9, %bb8, %bb
  store i32 1, ptr addrspace(3) null
  switch i32 0, label %bb7 [
    i32 0, label %bb8
  ]

bb7:                                              ; preds = %bb6
  unreachable

bb8:                                              ; preds = %bb6
  %tmp = phi ptr addrspace(5) [ null, %bb6 ]
  store i32 2, ptr addrspace(5) %tmp
  switch i32 %arg, label %bb6 [
    i32 0, label %bb10
    i32 1, label %bb9
  ]

bb9:                                              ; preds = %bb8
  br label %bb6

bb10:                                             ; preds = %bb8
  br label %bb6
}
