; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; CHECK: vmem
; CHECK: vmem
; CHECK-NOT:  r{{[0-9]*}} = add(r30,#-256)
; CHECK: vmem
; CHECK: vmem

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(i8* %a0, i8* %a1, i32 %a2, i8* %a3, i32 %a4) #0 {
b0:
  %v0 = alloca i8*, align 4
  %v1 = alloca i8*, align 4
  %v2 = alloca i32, align 4
  %v3 = alloca i8*, align 4
  %v4 = alloca i32, align 4
  %v5 = alloca <16 x i32>, align 64
  %v6 = alloca <32 x i32>, align 128
  store i8* %a0, i8** %v0, align 4
  store i8* %a1, i8** %v1, align 4
  store i32 %a2, i32* %v2, align 4
  store i8* %a3, i8** %v3, align 4
  store i32 %a4, i32* %v4, align 4
  %v7 = load i8*, i8** %v0, align 4
  %v8 = bitcast i8* %v7 to <16 x i32>*
  %v9 = load <16 x i32>, <16 x i32>* %v8, align 64
  %v10 = load i8*, i8** %v0, align 4
  %v11 = getelementptr inbounds i8, i8* %v10, i32 64
  %v12 = bitcast i8* %v11 to <16 x i32>*
  %v13 = load <16 x i32>, <16 x i32>* %v12, align 64
  %v14 = call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v9, <16 x i32> %v13)
  store <32 x i32> %v14, <32 x i32>* %v6, align 128
  %v15 = load i8*, i8** %v3, align 4
  %v16 = bitcast i8* %v15 to <16 x i32>*
  %v17 = load <16 x i32>, <16 x i32>* %v16, align 64
  store <16 x i32> %v17, <16 x i32>* %v5, align 64
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
