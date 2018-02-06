; RUN: llc -march=hexagon < %s | FileCheck %s

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

; --- add

; CHECK-LABEL: addb_64:
; CHECK: vadd(v0.b,v1.b)
define <64 x i8> @addb_64(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %p = add <64 x i8> %v0, %v1
  ret <64 x i8> %p
}

; CHECK-LABEL: addb_128:
; CHECK: vadd(v0.b,v1.b)
define <128 x i8> @addb_128(<128 x i8> %v0, <128 x i8> %v1) #1 {
  %p = add <128 x i8> %v0, %v1
  ret <128 x i8> %p
}

; CHECK-LABEL: addh_64:
; CHECK: vadd(v0.h,v1.h)
define <32 x i16> @addh_64(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %p = add <32 x i16> %v0, %v1
  ret <32 x i16> %p
}

; CHECK-LABEL: addh_128:
; CHECK: vadd(v0.h,v1.h)
define <64 x i16> @addh_128(<64 x i16> %v0, <64 x i16> %v1) #1 {
  %p = add <64 x i16> %v0, %v1
  ret <64 x i16> %p
}

; CHECK-LABEL: addw_64:
; CHECK: vadd(v0.w,v1.w)
define <16 x i32> @addw_64(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %p = add <16 x i32> %v0, %v1
  ret <16 x i32> %p
}

; CHECK-LABEL: addw_128:
; CHECK: vadd(v0.w,v1.w)
define <32 x i32> @addw_128(<32 x i32> %v0, <32 x i32> %v1) #1 {
  %p = add <32 x i32> %v0, %v1
  ret <32 x i32> %p
}

; --- sub

; CHECK-LABEL: subb_64:
; CHECK: vsub(v0.b,v1.b)
define <64 x i8> @subb_64(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %p = sub <64 x i8> %v0, %v1
  ret <64 x i8> %p
}

; CHECK-LABEL: subb_128:
; CHECK: vsub(v0.b,v1.b)
define <128 x i8> @subb_128(<128 x i8> %v0, <128 x i8> %v1) #1 {
  %p = sub <128 x i8> %v0, %v1
  ret <128 x i8> %p
}

; CHECK-LABEL: subh_64:
; CHECK: vsub(v0.h,v1.h)
define <32 x i16> @subh_64(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %p = sub <32 x i16> %v0, %v1
  ret <32 x i16> %p
}

; CHECK-LABEL: subh_128:
; CHECK: vsub(v0.h,v1.h)
define <64 x i16> @subh_128(<64 x i16> %v0, <64 x i16> %v1) #1 {
  %p = sub <64 x i16> %v0, %v1
  ret <64 x i16> %p
}

; CHECK-LABEL: subw_64:
; CHECK: vsub(v0.w,v1.w)
define <16 x i32> @subw_64(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %p = sub <16 x i32> %v0, %v1
  ret <16 x i32> %p
}

; CHECK-LABEL: subw_128:
; CHECK: vsub(v0.w,v1.w)
define <32 x i32> @subw_128(<32 x i32> %v0, <32 x i32> %v1) #1 {
  %p = sub <32 x i32> %v0, %v1
  ret <32 x i32> %p
}

; --- mul

; CHECK-LABEL: mpyb_64:
; CHECK: v[[H00:[0-9]+]]:[[L00:[0-9]+]].h = vmpy(v0.b,v1.b)
; CHECK: vshuffe(v[[H00]].b,v[[L00]].b)
define <64 x i8> @mpyb_64(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %p = mul <64 x i8> %v0, %v1
  ret <64 x i8> %p
}

; CHECK-LABEL: mpyb_128:
; CHECK: v[[H10:[0-9]+]]:[[L10:[0-9]+]].h = vmpy(v0.b,v1.b)
; CHECK: vshuffe(v[[H10]].b,v[[L10]].b)
define <128 x i8> @mpyb_128(<128 x i8> %v0, <128 x i8> %v1) #1 {
  %p = mul <128 x i8> %v0, %v1
  ret <128 x i8> %p
}

; CHECK-LABEL: mpyh_64:
; CHECK: vmpyi(v0.h,v1.h)
define <32 x i16> @mpyh_64(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %p = mul <32 x i16> %v0, %v1
  ret <32 x i16> %p
}

; CHECK-LABEL: mpyh_128:
; CHECK: vmpyi(v0.h,v1.h)
define <64 x i16> @mpyh_128(<64 x i16> %v0, <64 x i16> %v1) #1 {
  %p = mul <64 x i16> %v0, %v1
  ret <64 x i16> %p
}

; CHECK-LABEL: mpyw_64:
; CHECK-DAG: r[[T00:[0-9]+]] = #16
; CHECK-DAG: v[[T01:[0-9]+]].w = vmpyio(v0.w,v1.h)
; CHECK:     v[[T02:[0-9]+]].w = vasl(v[[T01]].w,r[[T00]])
; CHECK:     v[[T02]].w += vmpyie(v0.w,v1.uh)
define <16 x i32> @mpyw_64(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %p = mul <16 x i32> %v0, %v1
  ret <16 x i32> %p
}

; CHECK-LABEL: mpyw_128:
; CHECK-DAG: r[[T10:[0-9]+]] = #16
; CHECK-DAG: v[[T11:[0-9]+]].w = vmpyio(v0.w,v1.h)
; CHECK:     v[[T12:[0-9]+]].w = vasl(v[[T11]].w,r[[T10]])
; CHECK:     v[[T12]].w += vmpyie(v0.w,v1.uh)
define <32 x i32> @mpyw_128(<32 x i32> %v0, <32 x i32> %v1) #1 {
  %p = mul <32 x i32> %v0, %v1
  ret <32 x i32> %p
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
