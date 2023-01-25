; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: __hexagon_adddf3
; CHECK: __hexagon_subdf3

define void @f0(ptr %a0, double %a1, double %a2) #0 {
b0:
  %v0 = alloca ptr, align 4
  %v1 = alloca double, align 8
  %v2 = alloca double, align 8
  store ptr %a0, ptr %v0, align 4
  store double %a1, ptr %v1, align 8
  store double %a2, ptr %v2, align 8
  %v3 = load ptr, ptr %v0, align 4
  %v4 = load double, ptr %v3
  %v5 = load double, ptr %v1, align 8
  %v6 = fadd double %v4, %v5
  %v7 = load double, ptr %v2, align 8
  %v8 = fsub double %v6, %v7
  %v9 = load ptr, ptr %v0, align 4
  store double %v8, ptr %v9
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
