; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate new value stores.

; CHECK: r[[REG:[0-9]+]] = sfadd(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-NOT: }
; CHECK: memw({{.*}}) = r[[REG]].new
define void @f0(float %a0, float %a1) #0 {
b0:
  %v0 = alloca float, align 4
  %v1 = alloca float, align 4
  %v2 = alloca float*, align 4
  %v3 = alloca i32, align 4
  %v4 = load float, float* %v0, align 4
  %v5 = load float, float* %v1, align 4
  %v6 = fadd float %v5, %v4
  %v7 = load i32, i32* %v3, align 4
  %v8 = load float*, float** %v2, align 4
  %v9 = getelementptr inbounds float, float* %v8, i32 %v7
  store float %v6, float* %v9, align 4
  ret void
}

attributes #0 = { nounwind }
