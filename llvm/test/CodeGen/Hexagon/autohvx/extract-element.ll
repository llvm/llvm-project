; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that extract-element is handled.

; CHECK-LABEL: ext_00:
; CHECK:     r[[R000:[0-9]+]] = and(r0,#3)
; CHECK:     r[[R001:[0-9]+]] = vextract(v0,r0)
; CHECK-DAG: r[[R002:[0-9]+]] = asl(r[[R000]],#3)
; CHECK-DAG: r[[R003:[0-9]+]] = #8
; CHECK:                   r0 = extractu(r[[R001]],r[[R003]]:[[R002]])
define i8 @ext_00(<64 x i8> %a0, i32 %a1) #0 {
b2:
  %v3 = extractelement <64 x i8> %a0, i32 %a1
  ret i8 %v3
}

; CHECK-LABEL: ext_10:
; CHECK:     r[[R100:[0-9]+]] = and(r0,#3)
; CHECK:     r[[R101:[0-9]+]] = vextract(v0,r0)
; CHECK-DAG: r[[R102:[0-9]+]] = asl(r[[R100]],#3)
; CHECK-DAG: r[[R103:[0-9]+]] = #8
; CHECK:                   r0 = extractu(r[[R101]],r[[R103]]:[[R102]])
define i8 @ext_10(<128 x i8> %a0, i32 %a1) #1 {
b2:
  %v3 = extractelement <128 x i8> %a0, i32 %a1
  ret i8 %v3
}

; CHECK-LABEL: ext_01:
; CHECK-DAG: r[[R010:[0-9]+]] = asl(r0,#1)
; CHECK-DAG: r[[R011:[0-9]+]] = and(r0,#1)
; CHECK-DAG: r[[R012:[0-9]+]] = #16
; CHECK:     r[[R013:[0-9]+]] = asl(r[[R011]],#4)
; CHECK:     r[[R014:[0-9]+]] = vextract(v0,r[[R010]])
; CHECK:                   r0 = extractu(r[[R014]],r[[R012]]:[[R013]])
define i16 @ext_01(<32 x i16> %a0, i32 %a1) #0 {
b2:
  %v3 = extractelement <32 x i16> %a0, i32 %a1
  ret i16 %v3
}

; CHECK-LABEL: ext_11:
; CHECK-DAG: r[[R110:[0-9]+]] = asl(r0,#1)
; CHECK-DAG: r[[R111:[0-9]+]] = and(r0,#1)
; CHECK-DAG: r[[R112:[0-9]+]] = #16
; CHECK:     r[[R113:[0-9]+]] = asl(r[[R111]],#4)
; CHECK:     r[[R114:[0-9]+]] = vextract(v0,r[[R110]])
; CHECK:                   r0 = extractu(r[[R114]],r[[R112]]:[[R113]])
define i16 @ext_11(<64 x i16> %a0, i32 %a1) #1 {
b2:
  %v3 = extractelement <64 x i16> %a0, i32 %a1
  ret i16 %v3
}

; CHECK-LABEL: ext_02:
; CHECK: [[R020:r[0-9]+]] = asl(r0,#2)
; CHECK:               r0 = vextract(v0,[[R020]])
define i32 @ext_02(<16 x i32> %a0, i32 %a1) #0 {
b2:
  %v3 = extractelement <16 x i32> %a0, i32 %a1
  ret i32 %v3
}

; CHECK-LABEL: ext_12:
; CHECK: [[R120:r[0-9]+]] = asl(r0,#2)
; CHECK:               r0 = vextract(v0,[[R120]])
define i32 @ext_12(<32 x i32> %a0, i32 %a1) #1 {
b2:
  %v3 = extractelement <32 x i32> %a0, i32 %a1
  ret i32 %v3
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
