; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this doesn't crash. The select should be broken up into two
; vmux instructions.

; CHECK-LABEL: foo:
; CHECK: vmux
; CHECK: vmux
define <256 x i8> @foo(<256 x i8> %a0, <256 x i8> %a1) #0 {
  %v0 = icmp slt <256 x i8> %a0, zeroinitializer
  %v1 = select <256 x i1> %v0, <256 x i8> %a1, <256 x i8> %a0
  ret <256 x i8> %v1
}

attributes #0 = { "target-cpu"="hexagonv62" "target-features"="+hvx,+hvx-length128b" }
