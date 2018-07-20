; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK: q{{[0-3]}} = vsetq2(r{{[0-9]+}})

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
  %v4 = load <16 x i32>, <16 x i32>* %v2, align 64
  call void asm sideeffect "  $1 = vsetq2($0);\0A", "r,q"(i32 %v3, <16 x i32> %v4) #1
  ret void
}

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv62" "target-features"="+hvxv62,+hvx-length64b" }
attributes #1 = { nounwind }
