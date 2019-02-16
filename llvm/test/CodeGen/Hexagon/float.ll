; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: sfadd
; CHECK: sfsub

define void @f0(float* %a0, float %a1, float %a2) #0 {
b0:
  %v0 = alloca float*, align 4
  %v1 = alloca float, align 4
  %v2 = alloca float, align 4
  store float* %a0, float** %v0, align 4
  store float %a1, float* %v1, align 4
  store float %a2, float* %v2, align 4
  %v3 = load float*, float** %v0, align 4
  %v4 = load float, float* %v3
  %v5 = load float, float* %v1, align 4
  %v6 = fadd float %v4, %v5
  %v7 = load float, float* %v2, align 4
  %v8 = fsub float %v6, %v7
  %v9 = load float*, float** %v0, align 4
  store float %v8, float* %v9
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
