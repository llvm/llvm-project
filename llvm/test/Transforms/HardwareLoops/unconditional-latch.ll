; RUN: opt -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -hardware-loops -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ALLOW
; RUN: opt -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -hardware-loops -force-hardware-loop-guard=true -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ALLOW
; RUN: opt -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-hardware-loop-phi=true -hardware-loops -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LATCH

; CHECK-LABEL: not_rotated
; CHECK-LATCH-NOT: call void @llvm.set.loop.iterations
; CHECK-LATCH-NOT: call i1 @llvm.loop.decrement

; CHECK-ALLOW: bb:
; CHECK-ALLOW:   [[COUNT:%[^ ]+]] = add i32 %arg, 1
; CHECK-ALLOW:   br label %bb3
; CHECK-ALLOW: bb5:
; CHECK-ALLOW:   call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK-ALLOW:   br label %bb7

; CHECK-ALLOW: [[CMP:%[^ ]+]] = call i1 @llvm.loop.decrement.i32(i32 1)
; CHECK-ALLOW: br i1 [[CMP]], label %bb10, label %bb16
define void @not_rotated(i32 %arg, i16* nocapture %arg1, i16 signext %arg2) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb16, %bb
  %tmp = phi i32 [ 0, %bb ], [ %tmp17, %bb16 ]
  %tmp4 = icmp eq i32 %tmp, %arg
  br i1 %tmp4, label %bb18, label %bb5

bb5:                                              ; preds = %bb3
  %tmp6 = mul i32 %tmp, %arg
  br label %bb7

bb7:                                              ; preds = %bb10, %bb5
  %tmp8 = phi i32 [ %tmp15, %bb10 ], [ 0, %bb5 ]
  %tmp9 = icmp eq i32 %tmp8, %arg
  br i1 %tmp9, label %bb16, label %bb10

bb10:                                             ; preds = %bb7
  %tmp11 = add i32 %tmp8, %tmp6
  %tmp12 = getelementptr inbounds i16, i16* %arg1, i32 %tmp11
  %tmp13 = load i16, i16* %tmp12, align 2
  %tmp14 = add i16 %tmp13, %arg2
  store i16 %tmp14, i16* %tmp12, align 2
  %tmp15 = add i32 %tmp8, 1
  br label %bb7

bb16:                                             ; preds = %bb7
  %tmp17 = add i32 %tmp, 1
  br label %bb3

bb18:                                             ; preds = %bb3
  ret void
}
