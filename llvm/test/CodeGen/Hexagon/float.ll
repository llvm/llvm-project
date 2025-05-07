; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: sfadd
; CHECK: sfsub

define void @f0(ptr %a0, float %a1, float %a2) #0 {
b0:
  %v0 = alloca ptr, align 4
  %v1 = alloca float, align 4
  %v2 = alloca float, align 4
  store ptr %a0, ptr %v0, align 4
  store float %a1, ptr %v1, align 4
  store float %a2, ptr %v2, align 4
  %v3 = load ptr, ptr %v0, align 4
  %v4 = load float, ptr %v3
  %v5 = load float, ptr %v1, align 4
  %v6 = fadd float %v4, %v5
  %v7 = load float, ptr %v2, align 4
  %v8 = fsub float %v6, %v7
  %v9 = load ptr, ptr %v0, align 4
  store float %v8, ptr %v9
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
