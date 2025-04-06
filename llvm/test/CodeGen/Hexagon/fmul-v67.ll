; RUN: llc -mtriple=hexagon < %s | FileCheck %s


; CHECK-LABEL: test_00:
; CHECK: sfmpy(r0,r1)
define float @test_00(float %a0, float %a1) #0 {
b2:
  %v3 = fmul float %a0, %a1
  ret float %v3
}

; CHECK-LABEL: test_01:
; CHECK-DAG: [[R10:(r[0-9]+:[0-9]+)]] = dfmpyfix(r1:0,r3:2)
; CHECK-DAG: [[R11:(r[0-9]+:[0-9]+)]] = dfmpyfix(r3:2,r1:0)
; CHECK: [[R12:(r[0-9]+:[0-9]+)]] = dfmpyll([[R10]],[[R11]])
; CHECK: [[R12]] += dfmpylh([[R10]],[[R11]])
; CHECK: [[R12]] += dfmpylh([[R11]],[[R10]])
; CHECK: [[R12]] += dfmpyhh([[R10]],[[R11]])
define double @test_01(double %a0, double %a1) #1 {
b2:
  %v3 = fmul double %a0, %a1
  ret double %v3
}

; CHECK-LABEL: test_02:
; CHECK-DAG: [[R20:(r[0-9]+:[0-9]+)]] = dfmpyfix(r1:0,r3:2)
; CHECK-DAG: [[R21:(r[0-9]+:[0-9]+)]] = dfmpyfix(r3:2,r1:0)
; CHECK: [[R22:(r[0-9]+:[0-9]+)]] = dfmpyll([[R20]],[[R21]])
; CHECK: [[R22]] += dfmpylh([[R20]],[[R21]])
; CHECK: [[R22]] += dfmpylh([[R21]],[[R20]])
; CHECK: [[R22]] += dfmpyhh([[R20]],[[R21]])
define double @test_02(double %a0, double %a1) #2 {
b2:
  %v3 = fmul double %a0, %a1
  ret double %v3
}

; CHECK-LABEL: test_03:
; CHECK: [[R30:(r[0-9]+:[0-9]+)]] = dfmpyll(r1:0,r3:2)
; CHECK: [[R30]] += dfmpylh(r1:0,r3:2)
; CHECK: [[R30]] += dfmpylh(r3:2,r1:0)
; CHECK: [[R30]] += dfmpyhh(r1:0,r3:2)
define double @test_03(double %a0, double %a1) #3 {
b2:
  %v3 = fmul double %a0, %a1
  ret double %v3
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "target-cpu"="hexagonv67" }
attributes #2 = { nounwind "target-cpu"="hexagonv67" "unsafe-fp-math"="false" }
attributes #3 = { nounwind "target-cpu"="hexagonv67" "unsafe-fp-math"="true" }
