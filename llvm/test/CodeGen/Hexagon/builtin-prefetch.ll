; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: dcfetch
; CHECK: dcfetch{{.*}}#8
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define zeroext i8 @foo(i8* %addr) #0 {
entry:
  %addr.addr = alloca i8*, align 4
  store i8* %addr, i8** %addr.addr, align 4
  %0 = load i8*, i8** %addr.addr, align 4
  call void @llvm.prefetch(i8* %0, i32 0, i32 3, i32 1)
  %1 = load i8*, i8** %addr.addr, align 4
  %2 = bitcast i8* %1 to i32*
  %3 = load i32, i32* %2, align 4
  %4 = add i32 %3, 8
  %5 = inttoptr i32 %4 to i8*
  call void @llvm.hexagon.prefetch(i8* %5)
  %6 = load i8, i8* %5
  ret i8 %6
}

; Function Attrs: nounwind
declare void @llvm.prefetch(i8* nocapture, i32, i32, i32) #1
declare void @llvm.hexagon.prefetch(i8* nocapture) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
