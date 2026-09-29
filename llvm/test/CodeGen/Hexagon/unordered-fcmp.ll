; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that we generate correct set of instructions for unordered
; floating-point compares.

; CHECK-LABEL: f0:
; CHECK-DAG: [[PREG1:p[0-3]+]] = sfcmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: [[PREG2:p[0-3]+]] = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-3]+}} = or([[PREG2]],![[PREG1]])
define float @f0(float %a0, float %a1, float %a2) #0 {
b0:
  %v0 = fcmp une float %a0, 0.000000e+00
  %v1 = select i1 %v0, float %a2, float 0.000000e+00
  ret float %v1
}

; CHECK-LABEL: f1:
; CHECK-DAG: [[PREG1:p[0-3]+]] = sfcmp.ge(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: [[PREG2:p[0-3]+]] = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-3]+}} = or([[PREG2]],[[PREG1]])
define float @f1(float %a0, float %a1, float %a2) #0 {
b0:
  %v0 = fcmp uge float %a0, 0.000000e+00
  %v1 = select i1 %v0, float %a2, float 0.000000e+00
  ret float %v1
}

; CHECK-LABEL: f2:
; CHECK-DAG: [[PREG1:p[0-3]+]] = sfcmp.gt(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: [[PREG2:p[0-3]+]] = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-3]+}} = or([[PREG2]],[[PREG1]])
define float @f2(float %a0, float %a1, float %a2) #0 {
b0:
  %v0 = fcmp ugt float %a0, 0.000000e+00
  %v1 = select i1 %v0, float %a2, float 0.000000e+00
  ret float %v1
}

; CHECK-LABEL: f3:
; CHECK-DAG: [[PREG1:p[0-3]+]] = sfcmp.ge(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: [[PREG2:p[0-3]+]] = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-3]+}} = or([[PREG2]],[[PREG1]])
define float @f3(float %a0, float %a1, float %a2) #0 {
b0:
  %v0 = fcmp ule float %a0, 0.000000e+00
  %v1 = select i1 %v0, float %a2, float 0.000000e+00
  ret float %v1
}

; CHECK-LABEL: f4:
; CHECK-DAG: [[PREG1:p[0-3]+]] = sfcmp.gt(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: [[PREG2:p[0-3]+]] = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-3]+}} = or([[PREG2]],[[PREG1]])
define float @f4(float %a0, float %a1, float %a2) #0 {
b0:
  %v0 = fcmp ult float %a0, 0.000000e+00
  %v1 = select i1 %v0, float %a2, float 0.000000e+00
  ret float %v1
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
