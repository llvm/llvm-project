; RUN: llc -mtriple=x86_64-linux-gnu -stop-after=x86-isel %s -o - | llc --passes='print<machine-loops>' -x mir -o - 2>&1 | FileCheck %s

; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @foo(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %4, align 4
  br label %5

5:                                                ; preds = %13, %1
  %6 = load i32, ptr %4, align 4
  %7 = load i32, ptr %2, align 4
  %8 = icmp ne i32 %6, %7
  br i1 %8, label %9, label %16

9:                                                ; preds = %5
  %10 = load i32, ptr %4, align 4
  %11 = load i32, ptr %3, align 4
  %12 = add nsw i32 %11, %10
  store i32 %12, ptr %3, align 4
  br label %13

13:                                               ; preds = %9
  %14 = load i32, ptr %4, align 4
  %15 = add nsw i32 %14, 1
  store i32 %15, ptr %4, align 4
  br label %5, !llvm.loop !1

16:                                               ; preds = %5
  %17 = load i32, ptr %3, align 4
  %18 = load i32, ptr %2, align 4
  %19 = add nsw i32 %17, %18
  ret i32 %19
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}

; CHECK: Machine loop info for machine function 'foo':
; CHECK: Loop at depth 1 containing: %bb.1<header><exiting>,%bb.2,%bb.3<latch>
