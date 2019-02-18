; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: __hexagon_adddf3
; CHECK: __hexagon_subdf3

define void @f0(double* %a0, double %a1, double %a2) #0 {
b0:
  %v0 = alloca double*, align 4
  %v1 = alloca double, align 8
  %v2 = alloca double, align 8
  store double* %a0, double** %v0, align 4
  store double %a1, double* %v1, align 8
  store double %a2, double* %v2, align 8
  %v3 = load double*, double** %v0, align 4
  %v4 = load double, double* %v3
  %v5 = load double, double* %v1, align 8
  %v6 = fadd double %v4, %v5
  %v7 = load double, double* %v2, align 8
  %v8 = fsub double %v6, %v7
  %v9 = load double*, double** %v0, align 4
  store double %v8, double* %v9
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
