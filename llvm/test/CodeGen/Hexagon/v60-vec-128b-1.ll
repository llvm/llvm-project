; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK: v{{[0-9]+}} = vsplat(r{{[0-9]+}})
; CHECK: .comm g0,256,256
; CHECK: .comm g1,128,128

target triple = "hexagon"

@g0 = common global <64 x i32> zeroinitializer, align 256
@g1 = common global <32 x i32> zeroinitializer, align 128

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 0, i32* %v0
  %v1 = call i32 @f1(i8 zeroext 0)
  call void bitcast (void (...)* @f2 to void ()*)()
  %v2 = call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1)
  %v3 = call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 2)
  %v4 = call <64 x i32> @llvm.hexagon.V6.vaddubh.128B(<32 x i32> %v2, <32 x i32> %v3)
  %v5 = call <64 x i32> @llvm.hexagon.V6.vtmpyhb.128B(<64 x i32> %v4, i32 12)
  store <64 x i32> %v5, <64 x i32>* @g0, align 256
  call void @f3(i32 2048, i8* bitcast (<64 x i32>* @g0 to i8*))
  ret i32 0
}

declare i32 @f1(i8 zeroext) #0

declare void @f2(...) #0

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vtmpyhb.128B(<64 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vaddubh.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1

declare void @f3(i32, i8*) #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
