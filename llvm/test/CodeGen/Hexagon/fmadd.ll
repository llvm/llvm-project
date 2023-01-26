; RUN: llc -march=hexagon -fp-contract=fast < %s | FileCheck %s

@g0 = global float 0.000000e+00, align 4
@g1 = global float 1.000000e+00, align 4
@g2 = global float 2.000000e+00, align 4

; CHECK: r{{[0-9]+}} += sfmpy(r{{[0-9]+}},r{{[0-9]+}})
define void @f0() #0 {
b0:
  %v0 = load float, ptr @g0, align 4
  %v1 = load float, ptr @g1, align 4
  %v2 = load float, ptr @g2, align 4
  %v3 = alloca float, align 4
  %v4 = fmul float %v0, %v1
  %v5 = fadd float %v2, %v4
  store float %v5, ptr %v3, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
