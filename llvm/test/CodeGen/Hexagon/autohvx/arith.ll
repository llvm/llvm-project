; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; --- and

; CHECK-LABEL: andb_64:
; CHECK: vand(v0,v1)
define <64 x i8> @andb_64(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %p = and <64 x i8> %v0, %v1
  ret <64 x i8> %p
}

; CHECK-LABEL: andb_128:
; CHECK: vand(v0,v1)
define <128 x i8> @andb_128(<128 x i8> %v0, <128 x i8> %v1) #1 {
  %p = and <128 x i8> %v0, %v1
  ret <128 x i8> %p
}

; CHECK-LABEL: andh_64:
; CHECK: vand(v0,v1)
define <32 x i16> @andh_64(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %p = and <32 x i16> %v0, %v1
  ret <32 x i16> %p
}

; CHECK-LABEL: andh_128:
; CHECK: vand(v0,v1)
define <64 x i16> @andh_128(<64 x i16> %v0, <64 x i16> %v1) #1 {
  %p = and <64 x i16> %v0, %v1
  ret <64 x i16> %p
}

; CHECK-LABEL: andw_64:
; CHECK: vand(v0,v1)
define <16 x i32> @andw_64(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %p = and <16 x i32> %v0, %v1
  ret <16 x i32> %p
}

; CHECK-LABEL: andw_128:
; CHECK: vand(v0,v1)
define <32 x i32> @andw_128(<32 x i32> %v0, <32 x i32> %v1) #1 {
  %p = and <32 x i32> %v0, %v1
  ret <32 x i32> %p
}

; --- or

; CHECK-LABEL: orb_64:
; CHECK: vor(v0,v1)
define <64 x i8> @orb_64(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %p = or <64 x i8> %v0, %v1
  ret <64 x i8> %p
}

; CHECK-LABEL: orb_128:
; CHECK: vor(v0,v1)
define <128 x i8> @orb_128(<128 x i8> %v0, <128 x i8> %v1) #1 {
  %p = or <128 x i8> %v0, %v1
  ret <128 x i8> %p
}

; CHECK-LABEL: orh_64:
; CHECK: vor(v0,v1)
define <32 x i16> @orh_64(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %p = or <32 x i16> %v0, %v1
  ret <32 x i16> %p
}

; CHECK-LABEL: orh_128:
; CHECK: vor(v0,v1)
define <64 x i16> @orh_128(<64 x i16> %v0, <64 x i16> %v1) #1 {
  %p = or <64 x i16> %v0, %v1
  ret <64 x i16> %p
}

; CHECK-LABEL: orw_64:
; CHECK: vor(v0,v1)
define <16 x i32> @orw_64(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %p = or <16 x i32> %v0, %v1
  ret <16 x i32> %p
}

; CHECK-LABEL: orw_128:
; CHECK: vor(v0,v1)
define <32 x i32> @orw_128(<32 x i32> %v0, <32 x i32> %v1) #1 {
  %p = or <32 x i32> %v0, %v1
  ret <32 x i32> %p
}

; --- xor

; CHECK-LABEL: xorb_64:
; CHECK: vxor(v0,v1)
define <64 x i8> @xorb_64(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %p = xor <64 x i8> %v0, %v1
  ret <64 x i8> %p
}

; CHECK-LABEL: xorb_128:
; CHECK: vxor(v0,v1)
define <128 x i8> @xorb_128(<128 x i8> %v0, <128 x i8> %v1) #1 {
  %p = xor <128 x i8> %v0, %v1
  ret <128 x i8> %p
}

; CHECK-LABEL: xorh_64:
; CHECK: vxor(v0,v1)
define <32 x i16> @xorh_64(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %p = xor <32 x i16> %v0, %v1
  ret <32 x i16> %p
}

; CHECK-LABEL: xorh_128:
; CHECK: vxor(v0,v1)
define <64 x i16> @xorh_128(<64 x i16> %v0, <64 x i16> %v1) #1 {
  %p = xor <64 x i16> %v0, %v1
  ret <64 x i16> %p
}

; CHECK-LABEL: xorw_64:
; CHECK: vxor(v0,v1)
define <16 x i32> @xorw_64(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %p = xor <16 x i32> %v0, %v1
  ret <16 x i32> %p
}

; CHECK-LABEL: xorw_128:
; CHECK: vxor(v0,v1)
define <32 x i32> @xorw_128(<32 x i32> %v0, <32 x i32> %v1) #1 {
  %p = xor <32 x i32> %v0, %v1
  ret <32 x i32> %p
}

attributes #0 = { nounwind "target-cpu"="hexagonv73" "target-features"="+hvxv73,+hvx-length64b" }
attributes #1 = { nounwind "target-cpu"="hexagonv73" "target-features"="+hvxv73,+hvx-length128b" }
