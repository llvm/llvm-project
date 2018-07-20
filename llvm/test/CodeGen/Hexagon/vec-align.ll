; RUN: llc -O0 -march=hexagon < %s | FileCheck %s

; Make sure we generate stack alignment.
; CHECK: [[REG1:r[0-9]*]] = and(r29,#-64)
; CHECK: = add([[REG1]],#128)
; CHECK: = add([[REG1]],#64)
; Make sure we do not generate another -64 off SP.
; CHECK: vmem(
; CHECK-NOT: r{{[0-9]*}} = add(r29,#-64)

target triple = "hexagon"

@g0 = common global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca <16 x i32>, align 64
  %v2 = alloca <16 x i32>, align 64
  store i32 0, i32* %v0
  %v3 = call i32 @f1(i8 zeroext 0)
  %v4 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  store <16 x i32> %v4, <16 x i32>* %v1, align 64
  %v5 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 12)
  store <16 x i32> %v5, <16 x i32>* %v2, align 64
  %v6 = load <16 x i32>, <16 x i32>* %v1, align 64
  %v7 = load <16 x i32>, <16 x i32>* %v2, align 64
  %v8 = call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v6, <16 x i32> %v7)
  store <16 x i32> %v8, <16 x i32>* @g0, align 64
  call void bitcast (void (...)* @f2 to void ()*)()
  ret i32 0
}

declare i32 @f1(i8 zeroext) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #1

declare void @f2(...) #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
