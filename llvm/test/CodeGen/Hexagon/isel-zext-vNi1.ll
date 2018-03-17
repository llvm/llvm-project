; RUN: llc -march=hexagon -disable-hsdr < %s | FileCheck %s

; Check that zero-extends of short boolean vectors are done correctly.
; These are not the only possible instruction sequences, so if something
; changes, the tests should be changed as well.

; CHECK-LABEL: f0:
; CHECK-DAG: r[[D00:([0-9]+:[0-9]+)]] = combine(#0,r0)
; CHECK-DAG: r[[D01:([0-9]+:[0-9]+)]] = combine(#0,r1)
; CHECK: p[[P00:[0-3]]] = vcmpb.gt(r[[D01]],r[[D00]])
; CHECK: r{{[0-9]+}}:[[R00:[0-9]+]] = mask(p[[P00]])
; CHECK: r0 = and(r[[R00]],##16843009)
define <4 x i8> @f0(<4 x i8> %a0, <4 x i8> %a1) #0 {
b0:
  %v0 = icmp slt <4 x i8> %a0, %a1
  %v1 = zext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: f1:
; CHECK-DAG: r[[D10:([0-9]+:[0-9]+)]] = vsxthw(r0)
; CHECK-DAG: r[[D11:([0-9]+:[0-9]+)]] = vsxthw(r1)
; CHECK: p[[P10:[0-3]]] = vcmpw.gt(r[[D11]],r[[D10]])
; CHECK: r{{[0-9]+}}:[[R10:[0-9]+]] = mask(p[[P10]])
; CHECK: r0 = and(r[[R10]],##65537)
define <2 x i16> @f1(<2 x i16> %a0, <2 x i16> %a1) #0 {
b0:
  %v0 = icmp slt <2 x i16> %a0, %a1
  %v1 = zext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: f2:
; CHECK-DAG: r[[D20:([0-9]+:[0-9]+)]] = CONST64(#72340172838076673)
; CHECK-DAG: p[[P20:[0-3]]] = vcmpb.gt(r3:2,r1:0)
; CHECK: r[[D21:([0-9]+:[0-9]+)]] = mask(p[[P20]])
; CHECK: r1:0 = and(r[[D21]],r[[D20]])
define <8 x i8> @f2(<8 x i8> %a0, <8 x i8> %a1) #0 {
b0:
  %v0 = icmp slt <8 x i8> %a0, %a1
  %v1 = zext <8 x i1> %v0 to <8 x i8>
  ret <8 x i8> %v1
}

; CHECK-LABEL: f3:
; CHECK-DAG: r[[D30:([0-9]+:[0-9]+)]] = CONST64(#281479271743489)
; CHECK-DAG: p[[P30:[0-3]]] = vcmph.gt(r3:2,r1:0)
; CHECK: r[[D31:([0-9]+:[0-9]+)]] = mask(p[[P30]])
; CHECK: r1:0 = and(r[[D31]],r[[D30]])
define <4 x i16> @f3(<4 x i16> %a0, <4 x i16> %a1) #0 {
b0:
  %v0 = icmp slt <4 x i16> %a0, %a1
  %v1 = zext <4 x i1> %v0 to <4 x i16>
  ret <4 x i16> %v1
}

; CHECK-LABEL: f4:
; CHECK-DAG: r[[D40:([0-9]+:[0-9]+)]] = combine(#1,#1)
; CHECK-DAG: p[[P40:[0-3]]] = vcmpw.gt(r3:2,r1:0)
; CHECK: r[[D41:([0-9]+:[0-9]+)]] = mask(p[[P40]])
; CHECK: r1:0 = and(r[[D41]],r[[D40]])
define <2 x i32> @f4(<2 x i32> %a0, <2 x i32> %a1) #0 {
b0:
  %v0 = icmp slt <2 x i32> %a0, %a1
  %v1 = zext <2 x i1> %v0 to <2 x i32>
  ret <2 x i32> %v1
}

attributes #0 = { nounwind readnone }
