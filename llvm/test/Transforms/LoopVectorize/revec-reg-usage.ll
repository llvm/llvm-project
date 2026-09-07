; REQUIRES: asserts
; RUN: opt -disable-output -passes=loop-vectorize -vectorize-vector-loops \
; RUN:     -force-vector-width=1 -debug-only=vplan < %s 2>&1 \
; RUN:     | FileCheck %s
; RUN: opt -disable-output -passes=loop-vectorize -vectorize-vector-loops \
; RUN:     -force-vector-width="vscale x 1" -debug-only=vplan < %s 2>&1 \
; RUN:     | FileCheck %s

; When re-vectorising with VF = vscale x 1, the number of used registers is
; expected to remain the same, as we are turning fixed vectors into scalable
; ones and they belong to the same class according to the generic getRegisterClassForType.

; CHECK:      LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 3 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 2 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 1 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 1 registers
define void @register_usage(ptr noalias %dst, ptr noalias %src,
                            ptr noalias %threshold.ptr, i64 %n) {
entry:
  %threshold = load <4 x i32>, ptr %threshold.ptr, align 16
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %src.gep = getelementptr inbounds <4 x i32>, ptr %src, i64 %iv
  %dst.gep = getelementptr inbounds <4 x i32>, ptr %dst, i64 %iv
  %v = load <4 x i32>, ptr %src.gep, align 16
  %cmp = icmp sgt <4 x i32> %v, %threshold
  %sel = select <4 x i1> %cmp, <4 x i32> %v, <4 x i32> %threshold
  store <4 x i32> %sel, ptr %dst.gep, align 16
  %iv.next = add nuw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
