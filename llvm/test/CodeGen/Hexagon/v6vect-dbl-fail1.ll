; RUN: llc -mtriple=hexagon -O0 < %s | FileCheck %s
; CHECK: vmem
; CHECK: vmem
; CHECK-NOT:  r{{[0-9]*}} = add(r30,#-256)
; CHECK: vmem
; CHECK: vmem

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr %a0, ptr %a1, i32 %a2, ptr %a3, i32 %a4) #0 {
b0:
  %v0 = alloca ptr, align 4
  %v1 = alloca ptr, align 4
  %v2 = alloca i32, align 4
  %v3 = alloca ptr, align 4
  %v4 = alloca i32, align 4
  %v5 = alloca <16 x i32>, align 64
  %v6 = alloca <32 x i32>, align 128
  store ptr %a0, ptr %v0, align 4
  store ptr %a1, ptr %v1, align 4
  store i32 %a2, ptr %v2, align 4
  store ptr %a3, ptr %v3, align 4
  store i32 %a4, ptr %v4, align 4
  %v7 = load ptr, ptr %v0, align 4
  %v9 = load <16 x i32>, ptr %v7, align 64
  %v10 = load ptr, ptr %v0, align 4
  %v11 = getelementptr inbounds i8, ptr %v10, i32 64
  %v13 = load <16 x i32>, ptr %v11, align 64
  %v14 = call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v9, <16 x i32> %v13)
  store <32 x i32> %v14, ptr %v6, align 128
  %v15 = load ptr, ptr %v3, align 4
  %v17 = load <16 x i32>, ptr %v15, align 64
  store <16 x i32> %v17, ptr %v5, align 64
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
