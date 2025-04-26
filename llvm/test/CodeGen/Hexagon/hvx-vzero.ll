; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that V_vzero and W_vzero intrinsics work. The W_vzero intrinsic was added
; for v65/hvx.


; CHECK-LABEL: f0:
; CHECK: [[VREG1:v([0-9]+)]] = vxor([[VREG1]],[[VREG1]])
define void @f0(ptr nocapture %a0) #0 {
b0:
  %v1 = tail call <16 x i32> @llvm.hexagon.V6.vd0()
  store <16 x i32> %v1, ptr %a0, align 64
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vd0() #1

; CHECK-LABEL: f1:
; CHECK: [[VREG2:v([0-9]+):([0-9]+).w]] = vsub([[VREG2]],[[VREG2]])
define void @f1(ptr nocapture %a0) #0 {
b0:
  %v1 = tail call <32 x i32> @llvm.hexagon.V6.vdd0()
  store <32 x i32> %v1, ptr %a0, align 128
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdd0() #1

attributes #0 = { nounwind "target-cpu"="hexagonv65" "target-features"="+hvxv65,+hvx-length64b" }
attributes #1 = { nounwind readnone }
