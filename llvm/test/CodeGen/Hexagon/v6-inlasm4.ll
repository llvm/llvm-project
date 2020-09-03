; RUN: llc -march=hexagon -O2 -disable-hexagon-shuffle=1 < %s | FileCheck %s
; CHECK: q{{[0-3]}} = vsetq(r{{[0-9]+}})

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(i32 %a0, <16 x i32> %a1) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca <16 x i32>, align 64
  %v2 = alloca <16 x i32>, align 64
  store i32 %a0, i32* %v0, align 4
  store <16 x i32> %a1, <16 x i32>* %v1, align 64
  %v3 = load i32, i32* %v0, align 4
  %v4 = tail call <64 x i1> asm sideeffect "  $0 = vsetq($1);\0A", "=q,r"(i32 %v3) #1, !srcloc !0
  %v5 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v4, i32 -1)
  store <16 x i32> %v5, <16 x i32>* %v2, align 64
  ret void
}

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  ret i32 0
}

declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{i32 222}
