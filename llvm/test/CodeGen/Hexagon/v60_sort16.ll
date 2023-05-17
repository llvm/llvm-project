; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; Looking for 3rd register field to be restricted to r0-r7.
; v3:2=vdeal(v3,v2,r1)
; CHECK: v{{[0-9]+}}:{{[0-9]+}} = vdeal(v{{[0-9]+}},v{{[0-9]+}},r{{[0-7]+}})

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr %a0, i32 %a1, ptr %a2, ptr %a3) #0 {
b0:
  %v0 = alloca ptr, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca ptr, align 4
  %v3 = alloca ptr, align 4
  %v4 = alloca i32, align 4
  %v5 = alloca i32, align 4
  %v6 = alloca i32, align 4
  %v7 = alloca i32, align 4
  %v8 = alloca i32, align 4
  %v9 = alloca ptr, align 4
  %v10 = alloca ptr, align 4
  %v11 = alloca <16 x i32>, align 64
  %v12 = alloca <16 x i32>, align 64
  %v13 = alloca <32 x i32>, align 128
  %v14 = alloca <16 x i32>, align 64
  %v15 = alloca <16 x i32>, align 64
  %v16 = alloca <32 x i32>, align 128
  %v17 = alloca <16 x i32>, align 64
  %v18 = alloca <16 x i32>, align 64
  store ptr %a0, ptr %v0, align 4
  store i32 %a1, ptr %v1, align 4
  store ptr %a2, ptr %v2, align 4
  store ptr %a3, ptr %v3, align 4
  %v19 = load ptr, ptr %v2, align 4
  %v20 = getelementptr inbounds i8, ptr %v19, i32 192
  %v22 = load <16 x i32>, ptr %v20, align 64
  store <16 x i32> %v22, ptr %v12, align 64
  store i32 16843009, ptr %v4, align 4
  %v23 = load i32, ptr %v4, align 4
  %v24 = load i32, ptr %v4, align 4
  %v25 = add nsw i32 %v23, %v24
  store i32 %v25, ptr %v5, align 4
  %v26 = load i32, ptr %v5, align 4
  %v27 = load i32, ptr %v5, align 4
  %v28 = add nsw i32 %v26, %v27
  store i32 %v28, ptr %v6, align 4
  %v29 = load ptr, ptr %v0, align 4
  store ptr %v29, ptr %v9, align 4
  %v30 = load ptr, ptr %v3, align 4
  store ptr %v30, ptr %v10, align 4
  store i32 0, ptr %v8, align 4
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v31 = load i32, ptr %v8, align 4
  %v32 = load i32, ptr %v1, align 4
  %v33 = icmp slt i32 %v31, %v32
  br i1 %v33, label %b2, label %b4

b2:                                               ; preds = %b1
  %v34 = load <16 x i32>, ptr %v11, align 64
  %v35 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v34, i32 -1)
  %v36 = load <16 x i32>, ptr %v14, align 64
  %v37 = load <16 x i32>, ptr %v15, align 64
  %v38 = call <32 x i32> @llvm.hexagon.V6.vswap(<64 x i1> %v35, <16 x i32> %v36, <16 x i32> %v37)
  store <32 x i32> %v38, ptr %v13, align 128
  %v39 = load <32 x i32>, ptr %v13, align 128
  %v40 = call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v39)
  store <16 x i32> %v40, ptr %v14, align 64
  %v41 = load <32 x i32>, ptr %v13, align 128
  %v42 = call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v41)
  store <16 x i32> %v42, ptr %v15, align 64
  %v43 = load <16 x i32>, ptr %v17, align 64
  %v44 = load <16 x i32>, ptr %v18, align 64
  %v45 = load i32, ptr %v7, align 4
  %v46 = call <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32> %v43, <16 x i32> %v44, i32 %v45)
  store <32 x i32> %v46, ptr %v16, align 128
  br label %b3

b3:                                               ; preds = %b2
  %v47 = load i32, ptr %v8, align 4
  %v48 = add nsw i32 %v47, 1
  store i32 %v48, ptr %v8, align 4
  br label %b1

b4:                                               ; preds = %b1
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vswap(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
