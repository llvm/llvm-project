; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Check that we generate new value stores in V60.

; CHECK: v{{[0-9]+}} = valign(v{{[0-9]+}},v{{[0-9]+}},r{{[0-9]+}})
; CHECK: vmem(r{{[0-9]+}}+#{{[0-9]+}}) = v{{[0-9]+}}.new

define void @f0(i16* nocapture readonly %a0, i32 %a1, i16* nocapture %a2) #0 {
b0:
  %v0 = bitcast i16* %a0 to <16 x i32>*
  %v1 = bitcast i16* %a2 to <16 x i32>*
  %v2 = load <16 x i32>, <16 x i32>* %v0, align 64
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v2, <16 x i32> undef, i32 %a1)
  store <16 x i32> %v3, <16 x i32>* %v1, align 64
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
