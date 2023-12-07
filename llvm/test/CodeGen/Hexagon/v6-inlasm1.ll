; RUN: llc -march=hexagon -O2 -disable-hexagon-shuffle=1 < %s | FileCheck %s
; CHECK: vmemu(r{{[0-9]+}}+#0) = v{{[0-9]*}}

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr %a0, i32 %a1, ptr %a2, i32 %a3, ptr %a4) #0 {
b0:
  %v0 = alloca ptr, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca ptr, align 4
  %v3 = alloca i32, align 4
  %v4 = alloca ptr, align 4
  %v5 = alloca i32, align 4
  %v6 = alloca i32, align 4
  %v7 = alloca i32, align 4
  %v8 = alloca i32, align 4
  %v9 = alloca i32, align 4
  %v10 = alloca <16 x i32>, align 64
  %v11 = alloca <16 x i32>, align 64
  %v12 = alloca <16 x i32>, align 64
  %v13 = alloca <16 x i32>, align 64
  %v14 = alloca <16 x i32>, align 64
  %v15 = alloca <16 x i32>, align 64
  %v16 = alloca <16 x i32>, align 64
  %v17 = alloca <16 x i32>, align 64
  %v18 = alloca <16 x i32>, align 64
  %v19 = alloca <16 x i32>, align 64
  %v20 = alloca <16 x i32>, align 64
  store ptr %a0, ptr %v0, align 4
  store i32 %a1, ptr %v1, align 4
  store ptr %a2, ptr %v2, align 4
  store i32 %a3, ptr %v3, align 4
  store ptr %a4, ptr %v4, align 4
  %v21 = load i32, ptr %v1, align 4
  %v22 = ashr i32 %v21, 16
  %v23 = and i32 65535, %v22
  store i32 %v23, ptr %v8, align 4
  %v24 = load i32, ptr %v1, align 4
  %v25 = and i32 65535, %v24
  store i32 %v25, ptr %v5, align 4
  %v26 = load i32, ptr %v3, align 4
  %v27 = and i32 65535, %v26
  store i32 %v27, ptr %v6, align 4
  %v28 = load i32, ptr %v3, align 4
  %v29 = ashr i32 %v28, 16
  %v30 = and i32 65535, %v29
  store i32 %v30, ptr %v9, align 4
  %v31 = load ptr, ptr %v4, align 4
  %v33 = load <16 x i32>, ptr %v31, align 64
  store <16 x i32> %v33, ptr %v10, align 64
  %v34 = load ptr, ptr %v4, align 4
  %v35 = getelementptr inbounds i8, ptr %v34, i32 64
  %v37 = load <16 x i32>, ptr %v35, align 64
  store <16 x i32> %v37, ptr %v12, align 64
  %v38 = load i32, ptr %v9, align 4
  store i32 %v38, ptr %v7, align 4
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v39 = load i32, ptr %v7, align 4
  %v40 = icmp sge i32 %v39, 0
  br i1 %v40, label %b2, label %b4

b2:                                               ; preds = %b1
  %v41 = load ptr, ptr %v0, align 4
  %v43 = load <16 x i32>, ptr %v41, align 4
  store <16 x i32> %v43, ptr %v14, align 64
  %v44 = load i32, ptr %v5, align 4
  %v45 = load ptr, ptr %v0, align 4
  %v46 = getelementptr inbounds i8, ptr %v45, i32 %v44
  store ptr %v46, ptr %v0, align 4
  %v47 = load <16 x i32>, ptr %v14, align 64
  %v48 = load <16 x i32>, ptr %v10, align 64
  %v49 = call <16 x i32> @llvm.hexagon.V6.vrdelta(<16 x i32> %v47, <16 x i32> %v48)
  store <16 x i32> %v49, ptr %v15, align 64
  %v50 = load <16 x i32>, ptr %v14, align 64
  %v51 = load <16 x i32>, ptr %v12, align 64
  %v52 = call <16 x i32> @llvm.hexagon.V6.vrdelta(<16 x i32> %v50, <16 x i32> %v51)
  store <16 x i32> %v52, ptr %v17, align 64
  %v53 = load <16 x i32>, ptr %v15, align 64
  %v54 = load <16 x i32>, ptr %v17, align 64
  %v55 = call <16 x i32> @llvm.hexagon.V6.vavgub(<16 x i32> %v53, <16 x i32> %v54)
  store <16 x i32> %v55, ptr %v19, align 64
  %v56 = load ptr, ptr %v2, align 4
  %v57 = load <16 x i32>, ptr %v19, align 64
  call void asm sideeffect "  vmemu($0) = $1;\0A", "r,v,~{memory}"(ptr %v56, <16 x i32> %v57) #2, !srcloc !0
  br label %b3

b3:                                               ; preds = %b2
  %v58 = load i32, ptr %v6, align 4
  %v59 = load i32, ptr %v7, align 4
  %v60 = sub nsw i32 %v59, %v58
  store i32 %v60, ptr %v7, align 4
  br label %b1

b4:                                               ; preds = %b1
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrdelta(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavgub(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!0 = !{i32 1708}
