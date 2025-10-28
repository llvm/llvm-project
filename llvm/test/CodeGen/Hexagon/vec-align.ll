; RUN: llc -O0 -mtriple=hexagon < %s | FileCheck %s

; Make sure we generate stack alignment.
; CHECK: [[REG1:r[0-9]*]] = and(r29,#-64)
; CHECK: vmem([[REG1]]+#2) =
; CHECK: vmem([[REG1]]+#1) =
; CHECK: = vmem([[REG1]]+#2)
; CHECK: = vmem([[REG1]]+#1)

target triple = "hexagon"

@g0 = common global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca <16 x i32>, align 64
  %v2 = alloca <16 x i32>, align 64
  store i32 0, ptr %v0
  %v3 = call i32 @f1(i8 zeroext 0)
  %v4 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  store <16 x i32> %v4, ptr %v1, align 64
  %v5 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 12)
  store <16 x i32> %v5, ptr %v2, align 64
  %v6 = load <16 x i32>, ptr %v1, align 64
  %v7 = load <16 x i32>, ptr %v2, align 64
  %v8 = call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v6, <16 x i32> %v7)
  store <16 x i32> %v8, ptr @g0, align 64
  call void @f2()
  ret i32 0
}

declare i32 @f1(i8 zeroext) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #1

declare void @f2(...) #0

attributes #0 = { nounwind "target-cpu"="hexagonv65" "target-features"="+hvxv65,+hvx-length64b" }
attributes #1 = { nounwind readnone }
