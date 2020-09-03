; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that we compile the HVX dual output intrinsics.

; CHECK-LABEL: f0:
; CHECK: v{{[0-9]+}}.w = vadd(v{{[0-9]+}}.w,v{{[0-9]+}}.w,q{{[0-3]}}):carry
define inreg <32 x i32> @f0(<32 x i32> %a0, <32 x i32> %a1, i8* nocapture readonly %a2) #0 {
b0:
  %v0 = bitcast i8* %a2 to <32 x i32>*
  %v1 = load <32 x i32>, <32 x i32>* %v0, align 128
  %v2 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v1, i32 -1)
  %v3 = tail call { <32 x i32>, <128 x i1> } @llvm.hexagon.V6.vaddcarry.128B(<32 x i32> %a0, <32 x i32> %a1, <128 x i1> %v2)
  %v4 = extractvalue { <32 x i32>, <128 x i1> } %v3, 0
  ret <32 x i32> %v4
}

; CHECK-LABEL: f1:
; CHECK: v{{[0-9]+}}.w = vsub(v{{[0-9]+}}.w,v{{[0-9]+}}.w,q{{[0-3]}}):carry
define inreg <32 x i32> @f1(<32 x i32> %a0, <32 x i32> %a1, i8* nocapture readonly %a2) #0 {
b0:
  %v0 = bitcast i8* %a2 to <32 x i32>*
  %v1 = load <32 x i32>, <32 x i32>* %v0, align 128
  %v2 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v1, i32 -1)
  %v3 = tail call { <32 x i32>, <128 x i1> } @llvm.hexagon.V6.vsubcarry.128B(<32 x i32> %a0, <32 x i32> %a1, <128 x i1> %v2)
  %v4 = extractvalue { <32 x i32>, <128 x i1> } %v3, 0
  ret <32 x i32> %v4
}

; Function Attrs: nounwind readnone
declare { <32 x i32>, <128 x i1> } @llvm.hexagon.V6.vaddcarry.128B(<32 x i32>, <32 x i32>, <128 x i1>) #1

; Function Attrs: nounwind readnone
declare { <32 x i32>, <128 x i1> } @llvm.hexagon.V6.vsubcarry.128B(<32 x i32>, <32 x i32>, <128 x i1>) #1

; Function Attrs: nounwind readnone
declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv65" "target-features"="+hvxv65,+hvx-length128b" }
attributes #1 = { nounwind readnone }
