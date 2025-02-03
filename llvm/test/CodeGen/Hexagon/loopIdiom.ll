; RUN: opt -debug -S -march=hexagon -O2  < %s | FileCheck %s
; REQUIRES: asserts
; CHECK: define dso_local void @complexMultAccum
target triple = "hexagon"

; Function Attrs: noinline nounwind
define dso_local void @complexMultAccum(i32 noundef %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  %run_c_code = alloca i8, align 1
  %run_asm_code = alloca i8, align 1
  %iOutter = alloca i32, align 4
  %iOutter1 = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store i8 1, ptr %run_c_code, align 1
  store i8 0, ptr %run_asm_code, align 1
  %0 = load i8, ptr %run_c_code, align 1
  %tobool = icmp ne i8 %0, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %iOutter, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %1 = load i32, ptr %iOutter, align 4
  %cmp = icmp slt i32 %1, 2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %2 = load i32, ptr %iOutter, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %iOutter, align 4
  br label %for.cond, !llvm.loop !3

for.end:                                          ; preds = %for.cond
  store i32 0, ptr %iOutter1, align 4
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc5, %for.end
  %3 = load i32, ptr %iOutter1, align 4
  %cmp3 = icmp slt i32 %3, 2
  br i1 %cmp3, label %for.body4, label %for.end7

for.body4:                                        ; preds = %for.cond2
  br label %for.inc5

for.inc5:                                         ; preds = %for.body4
  %4 = load i32, ptr %iOutter1, align 4
  %inc6 = add nsw i32 %4, 1
  store i32 %inc6, ptr %iOutter1, align 4
  br label %for.cond2, !llvm.loop !5

for.end7:                                         ; preds = %for.cond2
  br label %if.end

if.end:                                           ; preds = %for.end7, %entry
  ret void
}

attributes #0 = { noinline nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+v79,-long-calls" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"LLVM Clang"}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.mustprogress"}
!5 = distinct !{!5, !4}

