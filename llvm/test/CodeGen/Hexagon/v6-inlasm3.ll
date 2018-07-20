; RUN: llc -march=hexagon -O2 -disable-hexagon-shuffle=1 < %s | FileCheck %s
; CHECK: vmemu(r{{[0-9]+}}) = v{{[0-9]*}}

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(i8* %a0, i8* %a1) #0 {
b0:
  %v0 = alloca i8*, align 4
  %v1 = alloca i8*, align 4
  %v2 = alloca <16 x i32>, align 64
  %v3 = alloca <16 x i32>, align 64
  %v4 = alloca <32 x i32>, align 128
  store i8* %a0, i8** %v0, align 4
  store i8* %a1, i8** %v1, align 4
  %v5 = load i8*, i8** %v0, align 4
  %v6 = load <16 x i32>, <16 x i32>* %v2, align 64
  call void asm sideeffect "  $1 = vmemu($0);\0A", "r,v"(i8* %v5, <16 x i32> %v6) #1, !srcloc !0
  %v7 = load i8*, i8** %v0, align 4
  %v8 = load <16 x i32>, <16 x i32>* %v3, align 64
  call void asm sideeffect "  $1 = vmemu($0);\0A", "r,v"(i8* %v7, <16 x i32> %v8) #1, !srcloc !1
  %v9 = load <32 x i32>, <32 x i32>* %v4, align 128
  %v10 = load <16 x i32>, <16 x i32>* %v2, align 64
  %v11 = load <16 x i32>, <16 x i32>* %v3, align 64
  call void asm sideeffect "  $0 = vcombine($1,$2);\0A", "v,v,v"(<32 x i32> %v9, <16 x i32> %v10, <16 x i32> %v11) #1, !srcloc !2
  %v12 = load i8*, i8** %v1, align 4
  %v13 = load <16 x i32>, <16 x i32>* %v2, align 64
  call void asm sideeffect "  vmemu($0) = $1;\0A", "r,v,~{memory}"(i8* %v12, <16 x i32> %v13) #1, !srcloc !3
  ret void
}

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind }

!0 = !{i32 272}
!1 = !{i32 348}
!2 = !{i32 424}
!3 = !{i32 519}
