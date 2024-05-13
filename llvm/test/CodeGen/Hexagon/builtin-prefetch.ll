; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: dcfetch
; CHECK: dcfetch{{.*}}#8
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define zeroext i8 @foo(ptr %addr) #0 {
entry:
  %addr.addr = alloca ptr, align 4
  store ptr %addr, ptr %addr.addr, align 4
  %0 = load ptr, ptr %addr.addr, align 4
  call void @llvm.prefetch(ptr %0, i32 0, i32 3, i32 1)
  %1 = load ptr, ptr %addr.addr, align 4
  %2 = load i32, ptr %1, align 4
  %3 = add i32 %2, 8
  %4 = inttoptr i32 %3 to ptr
  call void @llvm.hexagon.prefetch(ptr %4)
  %5 = load i8, ptr %4
  ret i8 %5
}

; Function Attrs: nounwind
declare void @llvm.prefetch(ptr nocapture, i32, i32, i32) #1
declare void @llvm.hexagon.prefetch(ptr nocapture) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
