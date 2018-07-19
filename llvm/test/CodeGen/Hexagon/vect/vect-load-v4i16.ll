; RUN: llc -march=hexagon -O0 < %s | FileCheck %s

; CHECK-LABEL: danny:
; CHECK-DAG: [[T0:r[0-9]+]] = memuh(r0+#0)
; CHECK-DAG: [[T1:r[0-9]+]] = memuh(r0+#2)
; CHECK:     [[T0]] |= asl([[T1]],#16)
; CHECK-DAG: [[T2:r[0-9]+]] = memuh(r0+#4)
; CHECK-DAG: [[T3:r[0-9]+]] = memuh(r0+#6)
; CHECK:     [[T2]] |= asl([[T3]],#16)
; CHECK:     combine([[T2]],[[T0]])
define <4 x i16> @danny(<4 x i16>* %p) {
  %t0 = load <4 x i16>, <4 x i16>* %p, align 2
  ret <4 x i16> %t0
}

; CHECK-LABEL: sammy:
; CHECK-DAG: [[T0:r[0-9]+]] = memw(r0+#0)
; CHECK-DAG: [[T1:r[0-9]+]] = memw(r0+#4)
; CHECK:     combine([[T1]],[[T0]])
define <4 x i16> @sammy(<4 x i16>* %p) {
  %t0 = load <4 x i16>, <4 x i16>* %p, align 4
  ret <4 x i16> %t0
}
