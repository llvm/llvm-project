; RUN: llc -march=hexagon -O0 -hexagon-align-loads=0 < %s | FileCheck %s

; CHECK-LABEL: danny:
; CHECK:     r1 = r0
; CHECK-DAG: [[T0:r[0-9]+]] = memuh(r1+#0)
; CHECK-DAG: [[T1:r[0-9]+]] = memuh(r1+#2)
; CHECK:     r2 |= asl([[T1]],#16)
; CHECK-DAG: [[T2:r[0-9]+]] = memuh(r1+#4)
; CHECK-DAG: [[T3:r[0-9]+]] = memuh(r1+#6)
; CHECK:     r1 |= asl([[T3]],#16)
define <4 x i16> @danny(ptr %p) {
  %t0 = load <4 x i16>, ptr %p, align 2
  ret <4 x i16> %t0
}

; CHECK-LABEL: sammy:
; CHECK-DAG: [[T0:r[0-9]+]] = memw(r0+#0)
; CHECK-DAG: r1 = memw(r0+#4)
; CHECK:     r0 = [[T0]]
define <4 x i16> @sammy(ptr %p) {
  %t0 = load <4 x i16>, ptr %p, align 4
  ret <4 x i16> %t0
}
