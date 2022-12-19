; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 < %s | FileCheck %s
; CHECK-NOT: CONST

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0(i64 %a0) #0 {
b0:
  %v0 = alloca i64, align 8
  store i64 %a0, ptr %v0, align 8
  %v1 = call i32 @llvm.hexagon.S2.ct0p(i64 4222189076152335)
  ret i32 %v1
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.ct0p(i64) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
