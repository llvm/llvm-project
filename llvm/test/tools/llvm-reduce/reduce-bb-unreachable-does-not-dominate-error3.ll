; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=unreachable-basic-blocks,basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s < %t


; CHECK-INTERESTINGNESS: store i32 0,
; CHECK-INTERESTINGNESS: store i32 1,


; CHECK: bb:
; CHECK-NEXT: %tmp = icmp eq i8 0, 0
; CHECK-NEXT: %tmp12 = load i32, ptr addrspace(4) inttoptr (i64 3016 to ptr addrspace(4)), align 8
; CHECK-NEXT: %tmp13 = load i32, ptr addrspace(4) null, align 8
; CHECK-NEXT: br label %bb20

; CHECK: bb20:
; CHECK-NEXT: store i32 0, ptr addrspace(3) null, align 4
; CHECK-NEXT: br label %bb21

; CHECK: bb21:
; CHECK-NEXT: store i32 1, ptr addrspace(3) null, align 4
; CHECK-NEXT: ret void

define void @snork() {
bb:
  %tmp = icmp eq i8 0, 0
  %tmp12 = load i32, ptr addrspace(4) inttoptr (i64 3016 to ptr addrspace(4)), align 8
  %tmp13 = load i32, ptr addrspace(4) null, align 8
  br label %bb14

bb14:                                             ; preds = %bb21, %bb20, %bb19, %bb14, %bb
  switch i32 %tmp12, label %bb22 [
    i32 2, label %bb14
    i32 1, label %bb19
  ]

bb15:                                             ; preds = %bb17
  %tmp16 = fadd contract double %tmp18, 1.000000e+00
  unreachable

bb17:                                             ; preds = %bb17
  %tmp18 = fadd contract double 0.000000e+00, 0.000000e+00
  br i1 false, label %bb15, label %bb17

bb19:                                             ; preds = %bb14
  switch i32 %tmp13, label %bb14 [
    i32 2, label %bb21
    i32 1, label %bb20
  ]

bb20:                                             ; preds = %bb19
  store i32 0, ptr addrspace(3) null, align 4
  br label %bb14

bb21:                                             ; preds = %bb19
  store i32 1, ptr addrspace(3) null, align 4
  br label %bb14

bb22:                                             ; preds = %bb14
  unreachable
}
