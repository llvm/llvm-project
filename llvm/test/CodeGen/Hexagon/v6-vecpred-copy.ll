; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; CHECK: v{{[0-9]*}} = vxor(v{{[0-9]*}},v{{[0-9]*}})
; CHECK: if (q{{[-0-3]}}) v{{[0-9]*}}.b += v{{[0-9]*}}.b
; CHECK: if (q{{[-0-3]}}) v{{[0-9]*}}.b -= v{{[0-9]*}}.b
; CHECK: if (q{{[-0-3]}}) v{{[0-9]*}}.h += v{{[0-9]*}}.h
; CHECK: if (q{{[-0-3]}}) v{{[0-9]*}}.h -= v{{[0-9]*}}.h
; CHECK: if (q{{[-0-3]}}) v{{[0-9]*}}.w += v{{[0-9]*}}.w
; CHECK: if (q{{[-0-3]}}) v{{[0-9]*}}.w -= v{{[0-9]*}}.w
; CHECK: if (!q{{[-0-3]}}) v{{[0-9]*}}.b += v{{[0-9]*}}.b
; CHECK: if (!q{{[-0-3]}}) v{{[0-9]*}}.b -= v{{[0-9]*}}.b
; CHECK: if (!q{{[-0-3]}}) v{{[0-9]*}}.h += v{{[0-9]*}}.h
; CHECK: if (!q{{[-0-3]}}) v{{[0-9]*}}.h -= v{{[0-9]*}}.h
; CHECK: if (!q{{[-0-3]}}) v{{[0-9]*}}.w += v{{[0-9]*}}.w
; CHECK: if (!q{{[-0-3]}}) v{{[0-9]*}}.w -= v{{[0-9]*}}.w

target triple = "hexagon"

@g0 = common global <16 x i32> zeroinitializer, align 64
@g1 = common global <16 x i32> zeroinitializer, align 64
@g2 = common global <16 x i32> zeroinitializer, align 64
@g3 = common global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = call <16 x i32> @llvm.hexagon.V6.vd0()
  store <16 x i32> %v0, <16 x i32>* @g0, align 64
  %v1 = call <16 x i32> @llvm.hexagon.V6.vd0()
  store <16 x i32> %v1, <16 x i32>* @g1, align 64
  %v2 = call <16 x i32> @llvm.hexagon.V6.vd0()
  store <16 x i32> %v2, <16 x i32>* @g2, align 64
  %v3 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v4 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v3, i32 -1)
  %v5 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v6 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v7 = call <16 x i32> @llvm.hexagon.V6.vaddbq(<64 x i1> %v4, <16 x i32> %v5, <16 x i32> %v6)
  store <16 x i32> %v7, <16 x i32>* @g2, align 64
  %v8 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v9 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v8, i32 -1)
  %v10 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v11 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v12 = call <16 x i32> @llvm.hexagon.V6.vsubbq(<64 x i1> %v9, <16 x i32> %v10, <16 x i32> %v11)
  store <16 x i32> %v12, <16 x i32>* @g2, align 64
  %v13 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v14 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v13, i32 -1)
  %v15 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v16 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v17 = call <16 x i32> @llvm.hexagon.V6.vaddhq(<64 x i1> %v14, <16 x i32> %v15, <16 x i32> %v16)
  store <16 x i32> %v17, <16 x i32>* @g2, align 64
  %v18 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v19 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v18, i32 -1)
  %v20 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v21 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v22 = call <16 x i32> @llvm.hexagon.V6.vsubhq(<64 x i1> %v19, <16 x i32> %v20, <16 x i32> %v21)
  store <16 x i32> %v22, <16 x i32>* @g2, align 64
  %v23 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v24 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v23, i32 -1)
  %v25 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v26 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v27 = call <16 x i32> @llvm.hexagon.V6.vaddwq(<64 x i1> %v24, <16 x i32> %v25, <16 x i32> %v26)
  store <16 x i32> %v27, <16 x i32>* @g2, align 64
  %v28 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v29 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v28, i32 -1)
  %v30 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v31 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v32 = call <16 x i32> @llvm.hexagon.V6.vsubwq(<64 x i1> %v29, <16 x i32> %v30, <16 x i32> %v31)
  store <16 x i32> %v32, <16 x i32>* @g2, align 64
  %v33 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v34 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v33, i32 -1)
  %v35 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v36 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v37 = call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v34, <16 x i32> %v35, <16 x i32> %v36)
  store <16 x i32> %v37, <16 x i32>* @g2, align 64
  %v38 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v39 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v38, i32 -1)
  %v40 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v41 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v42 = call <16 x i32> @llvm.hexagon.V6.vsubbnq(<64 x i1> %v39, <16 x i32> %v40, <16 x i32> %v41)
  store <16 x i32> %v42, <16 x i32>* @g2, align 64
  %v43 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v44 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v43, i32 -1)
  %v45 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v46 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v47 = call <16 x i32> @llvm.hexagon.V6.vaddhnq(<64 x i1> %v44, <16 x i32> %v45, <16 x i32> %v46)
  store <16 x i32> %v47, <16 x i32>* @g2, align 64
  %v48 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v49 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v48, i32 -1)
  %v50 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v51 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v52 = call <16 x i32> @llvm.hexagon.V6.vsubhnq(<64 x i1> %v49, <16 x i32> %v50, <16 x i32> %v51)
  store <16 x i32> %v52, <16 x i32>* @g2, align 64
  %v53 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v54 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v53, i32 -1)
  %v55 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v56 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v57 = call <16 x i32> @llvm.hexagon.V6.vaddwnq(<64 x i1> %v54, <16 x i32> %v55, <16 x i32> %v56)
  store <16 x i32> %v57, <16 x i32>* @g2, align 64
  %v58 = load <16 x i32>, <16 x i32>* @g3, align 64
  %v59 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v58, i32 -1)
  %v60 = load <16 x i32>, <16 x i32>* @g2, align 64
  %v61 = load <16 x i32>, <16 x i32>* @g1, align 64
  %v62 = call <16 x i32> @llvm.hexagon.V6.vsubwnq(<64 x i1> %v59, <16 x i32> %v60, <16 x i32> %v61)
  store <16 x i32> %v62, <16 x i32>* @g2, align 64
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vd0() #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddbq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubbq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddhq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubhq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddwq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubwq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubbnq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddhnq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubhnq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddwnq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubwnq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
