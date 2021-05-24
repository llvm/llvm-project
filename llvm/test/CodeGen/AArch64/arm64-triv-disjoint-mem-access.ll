; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a53 -enable-aa-sched-mi | FileCheck %s
; Check that the scheduler moves the load from a[1] past the store into a[2].
@a = common global i32* null, align 8
@m = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @func(i32 %i, i32 %j, i32 %k) #0 {
entry:
; CHECK: ldr {{w[0-9]+}}, [x[[REG:[0-9]+]], #4]
; CHECK: str {{w[0-9]+}}, [x[[REG]], #8]
  %0 = load i32*, i32** @a, align 8, !tbaa !1
  %arrayidx = getelementptr inbounds i32, i32* %0, i64 2
  store i32 %i, i32* %arrayidx, align 4, !tbaa !5
  %arrayidx1 = getelementptr inbounds i32, i32* %0, i64 1
  %1 = load i32, i32* %arrayidx1, align 4, !tbaa !5
  %add = add nsw i32 %k, %i
  store i32 %add, i32* @m, align 4, !tbaa !5
  ret i32 %1
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.6.0 "}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}
