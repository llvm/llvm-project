; RUN: llc -march=hexagon -hexagon-gen-mux-threshold=0 < %s | FileCheck %s --check-prefix=CHECK0
; RUN: llc -march=hexagon -hexagon-gen-mux-threshold=4 < %s | FileCheck %s --check-prefix=CHECK4

; Generate mux with threshold = 0:
; CHECK0: [[R0:r[0-9]+]] = add(r0,#-48)
; CHECK0: [[P0:p[0-3]]] = cmpb.gtu([[R0]],#9)
; CHECK0: r0 = mux([[P0]],#0,#1)

; No mux for threshold = 4:
; CHECK4-NOT: mux

define zeroext i8 @f0(i8 zeroext %a0) #0 {
b0:
  %v0 = add i8 %a0, -48
  %v1 = icmp ult i8 %v0, 10
  %v2 = zext i1 %v1 to i8
  ret i8 %v2
}

attributes #0 = { nounwind readnone }
