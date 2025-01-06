; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon"

@x = common global ptr null, align 4

; Function Attrs: nounwind
define i32 @inotify_init() #0 {
entry:
  %0 = tail call i32 asm sideeffect "trap0(#1);\0A", "={r0},{r6},~{memory}"(i32 1043) #1, !srcloc !1
  %cmp = icmp sgt i32 %0, -4096
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %sub = sub nsw i32 0, %0
  %1 = load ptr, ptr @x, align 4, !tbaa !2
  store i32 %sub, ptr %1, align 4, !tbaa !6
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %retval1.0 = phi i32 [ -1, %if.then ], [ %0, %entry ]
  ret i32 %retval1.0
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!1 = !{i32 155}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !4, i64 0}
