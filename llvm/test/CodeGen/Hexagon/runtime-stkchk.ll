; RUN: llc -mtriple=hexagon -mcpu=hexagonv55 -enable-stackovf-sanitizer < %s | FileCheck %s

; CHECK-LABEL: foo_1
; CHECK: __runtime_stack_check
define i32 @foo_1(i32 %n) #0 {
entry:
  %local = alloca [1024 x i32], align 8
  call void @llvm.lifetime.start.p0(i64 4096, ptr %local) #1
  call void @baz_1(ptr %local) #3
  %arrayidx = getelementptr inbounds [1024 x i32], ptr %local, i32 0, i32 %n
  %0 = load i32, ptr %arrayidx, align 4
  call void @llvm.lifetime.end.p0(i64 4096, ptr %local) #1
  ret i32 %0
}

; CHECK-LABEL: foo_2
; CHECK: __save_r16_through_r19_stkchk
define i32 @foo_2(i32 %n, ptr %y) #0 {
entry:
  %local = alloca [2048 x i32], align 8
  call void @llvm.lifetime.start.p0(i64 8192, ptr %local) #1
  call void @baz_2(ptr %y, ptr %local) #3
  %0 = load i32, ptr %y, align 4
  %add = add nsw i32 %n, %0
  %arrayidx = getelementptr inbounds [2048 x i32], ptr %local, i32 0, i32 %add
  %1 = load i32, ptr %arrayidx, align 4
  call void @llvm.lifetime.end.p0(i64 8192, ptr %local) #1
  ret i32 %1
}

declare void @baz_1(ptr) #2
declare void @baz_2(ptr, ptr) #2
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { optsize }

