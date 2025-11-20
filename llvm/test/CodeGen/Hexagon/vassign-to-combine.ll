; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; This testcase is known to generate an opportunity for creating vcombine
; in HexagonCopyToCombine.

; CHECK: vcombine

target triple = "hexagon-unknown--elf"

declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32>, <64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32>, <64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vaddubh.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vmpyub.128B(<32 x i32>, i32) #0

define void @f0(ptr %a0, ptr %a1) local_unnamed_addr #1 {
b0:
  %v0 = load <32 x i32>, ptr %a0, align 128
  %v1 = load <32 x i32>, ptr %a1, align 128
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  %v2 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> %v0, <32 x i32> %v1, i32 1)
  %v3 = tail call <64 x i32> @llvm.hexagon.V6.vmpyub.128B(<32 x i32> %v2, i32 33686018) #1
  %v4 = tail call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32> undef, <64 x i32> %v3) #1
  %v5 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v4)
  %v6 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32> %v5, <32 x i32> undef) #1
  %v7 = tail call <64 x i32> @llvm.hexagon.V6.vaddubh.128B(<32 x i32> %v6, <32 x i32> undef)
  %v8 = tail call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32> undef, <64 x i32> %v7) #1
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v8) #1
  %v10 = tail call <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32> %v9, <32 x i32> undef) #1
  store <32 x i32> %v10, ptr %a0, align 128
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v11 = phi <32 x i32> [ zeroinitializer, %b1 ], [ %v0, %b0 ]
  %v12 = phi <32 x i32> [ %v0, %b1 ], [ %v1, %b0 ]
  %v13 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> %v11, <32 x i32> %v12, i32 1)
  %v14 = tail call <64 x i32> @llvm.hexagon.V6.vmpyub.128B(<32 x i32> %v13, i32 33686018) #1
  %v15 = tail call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32> undef, <64 x i32> %v14) #1
  %v16 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v15)
  %v17 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32> %v16, <32 x i32> undef) #1
  %v18 = tail call <64 x i32> @llvm.hexagon.V6.vaddubh.128B(<32 x i32> %v17, <32 x i32> undef)
  %v19 = tail call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32> undef, <64 x i32> %v18) #1
  %v20 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v19) #1
  %v21 = tail call <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32> %v20, <32 x i32> undef) #1
  store <32 x i32> %v21, ptr %a1, align 128
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
