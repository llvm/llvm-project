; RUNNING TEST EXAMPLE: llvm-lit llvm/test/Transforms/Yk/BasicBlockTracer.ll
; RUN: llc -stop-after yk-basicblock-tracer-pass --yk-basicblock-tracer < %s  | FileCheck %s

; CHECK-LABEL: define dso_local noundef i32 @main()
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 0, i32 0)
define dso_local noundef i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 0, i32* %2, align 4
  store i32 0, i32* %3, align 4
  br label %4

; CHECK-LABEL: 4:{{.*}}
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 0, i32 1)
4:                                                ; preds = %13, %0
  %5 = load i32, i32* %3, align 4
  %6 = icmp slt i32 %5, 43
  br i1 %6, label %7, label %16

; CHECK-LABEL: 7:{{.*}}
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 0, i32 2)
7:                                                ; preds = %4
  %8 = load i32, i32* %3, align 4
  %9 = icmp eq i32 %8, 42
  br i1 %9, label %10, label %12

; CHECK-LABEL: 10:{{.*}}
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 0, i32 3)
10:                                               ; preds = %7
  %11 = load i32, i32* %3, align 4
  store i32 %11, i32* %1, align 4
  br label %17

; CHECK-LABEL: 12:{{.*}}
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 0, i32 4)
12:                                               ; preds = %7
  br label %13

; CHECK-LABEL: 13:{{.*}}
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 0, i32 5)
13:                                               ; preds = %12
  %14 = load i32, i32* %3, align 4
  %15 = add nsw i32 %14, 1
  store i32 %15, i32* %3, align 4
  br label %4, !llvm.loop !6

; CHECK-LABEL: 16:{{.*}}
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 0, i32 6)
16:                                               ; preds = %4
  store i32 0, i32* %1, align 4
  br label %17

; CHECK-LABEL: 17:{{.*}}
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 0, i32 7)
17:                                               ; preds = %16, %10
  %18 = load i32, i32* %1, align 4
  ret i32 %18
}

; CHECK-LABEL: define dso_local noundef i32 @_Z5checki(i32 noundef %0)
; CHECK-NEXT:  call void @yk_trace_basicblock(i32 1, i32 0)
define dso_local noundef i32 @_Z5checki(i32 noundef %0) #1 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = icmp eq i32 %3, 42
  %5 = zext i1 %4 to i32
  ret i32 %5
}

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Debian clang version 14.0.6"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
