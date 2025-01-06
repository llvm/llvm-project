; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that vector is produced with vxor
; CHECK: v{{[0-9]*}} = vxor
define <32 x i32> @f0(i32 %x) #0 {
  %vect = insertelement <32 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %x, i32 0
  ret <32 x i32> %vect
}

; Check that vector is produced with vsplat
; CHECK: v{{[0-9]*}} = vsplat
define <32 x i32> @f1(i32 %x) #0 {
  %vect = insertelement <32 x i32> <i32 undef, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, i32 %x, i32 0
  ret <32 x i32> %vect
}

; Check that the correct vror is generated
; CHECK: [[REG0:r([0-9]+)]] = #120
; CHECK: vror(v{{[0-9]+}},[[REG0]])
define <32 x i32> @f2(i32 %x) #0 {
  %vect = insertelement <32 x i32> <i32 1, i32 1, i32 undef, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, i32 %x, i32 2
  ret <32 x i32> %vect
}

; Check that the correct vror is generated
; CHECK: [[REG0:r([0-9]+)]] = #12
; CHECK: vror(v{{[0-9]+}},[[REG0]])
define <32 x i32> @f3(i32 %x) #0 {
  %vect = insertelement <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 undef, i32 1, i32 1>, i32 %x, i32 29
  ret <32 x i32> %vect
}

attributes #0 = { readnone nounwind "target-cpu"="hexagonv62" "target-features"="+hvx,+hvx-length128b" }

