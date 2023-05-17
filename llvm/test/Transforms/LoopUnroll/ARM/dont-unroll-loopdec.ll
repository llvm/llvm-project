; RUN: opt -mtriple=thumbv8.1m.main -mattr=+mve.fp -passes=loop-unroll -S < %s | FileCheck %s

; CHECK-LABEL:  foo
; CHECK:        5:
; CHECK:        6:                 ; preds = %6, %5
; CHECK:        15:                ; preds = %6
; CHECK:          br label %16
; CHECK:        16:                ; preds = %15, %3
; CHECK:          ret void
; CHECK:        }

define void @foo(ptr nocapture, ptr nocapture readonly, i32) {
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %16

; <label>:5:
  br label %6

; <label>:6:
  %7 = phi i32 [ %13, %6 ], [ %2, %5 ]
  %8 = phi ptr [ %10, %6 ], [ %1, %5 ]
  %9 = phi ptr [ %12, %6 ], [ %0, %5 ]
  %10 = getelementptr inbounds i8, ptr %8, i32 1
  %11 = load i8, ptr %8, align 1
  %12 = getelementptr inbounds i8, ptr %9, i32 1
  store i8 %11, ptr %9, align 1

  %13 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %7, i32 1)

  %14 = icmp sgt i32 %7, 1
  br i1 %14, label %6, label %15

; <label>:15:
  br label %16

; <label>:16:
  ret void
}

declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32)
