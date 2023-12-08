; RUN: llc -march=hexagon < %s | FileCheck %s

; Make sure that element no.1 extracted from <2 x i1> translates to extracting
; bit no.4 from the predicate register.

; CHECK: p[[P0:[0-3]]] = vcmpw.eq(r1:0,r3:2)
; CHECK: r[[R0:[0-9]+]] = p[[P0]]
; This is what we're really testing: the bit index of 4.
; CHECK: p[[P0]] = tstbit(r[[R0]],#4)

define i32 @fred(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp eq <2 x i32> %a0, %a1
  %v1 = extractelement <2 x i1> %v0, i32 1
  %v2 = zext i1 %v1 to i32
  ret i32 %v2
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
