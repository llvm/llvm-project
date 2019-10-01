; RUN: opt -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -hardware-loops -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ALLOW
; RUN: opt -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-hardware-loop-phi=true -hardware-loops -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LATCH

; CHECK-LABEL: not_rotated
; CHECK-LATCH-NOT: call void @llvm.set.loop.iterations
; CHECK-LATCH-NOT: call i1 @llvm.loop.decrement

; CHECK-ALLOW: call void @llvm.set.loop.iterations.i32(i32 %4)
; CHECK-ALLOW: br label %10

; CHECK-ALLOW: [[CMP:%[^ ]+]] = call i1 @llvm.loop.decrement.i32(i32 1)
; CHECK-ALLOW: br i1 [[CMP]], label %13, label %19

define void @not_rotated(i32, i16* nocapture, i16 signext) {
  br label %4

4:
  %5 = phi i32 [ 0, %3 ], [ %19, %18 ]
  %6 = icmp eq i32 %5, %0
  br i1 %6, label %20, label %7

7:
  %8 = mul i32 %5, %0
  br label %9

9:
  %10 = phi i32 [ %17, %12 ], [ 0, %7 ]
  %11 = icmp eq i32 %10, %0
  br i1 %11, label %18, label %12

12:
  %13 = add i32 %10, %8
  %14 = getelementptr inbounds i16, i16* %1, i32 %13
  %15 = load i16, i16* %14, align 2
  %16 = add i16 %15, %2
  store i16 %16, i16* %14, align 2
  %17 = add i32 %10, 1
  br label %9

18:
  %19 = add i32 %5, 1
  br label %4

20:
  ret void
}
